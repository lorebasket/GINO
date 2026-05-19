# Capytaine - BEM Radiation Precomputation

This folder contains the Capytaine-based Boundary Element Method (BEM) workflow used to
compute hydroelastic added-mass and radiation-damping matrices for the flutter solvers.
PanelAero still provides the circulatory DLM/VLM generalized force matrices; Capytaine
adds the non-circulatory BEM contribution from the submerged hydrofoil mesh.

The two main scripts are:

- `build_capytaine_mesh.py`: builds the wetted-body surface mesh used by Capytaine.
- `run_modal_radiation.py`: solves radiation problems and writes modal added mass
  `A(omega)` and radiation damping `B(omega)`.

## Dependencies

Install the BEM-specific packages in the same Python environment used for the FSI runs:

```bash
pip install capytaine gmsh meshio
```

`vtk` is optional and is only needed for the interactive mesh preview. Without it, use
`--matplotlib` or open the exported `.vtu` mesh in ParaView.

## Required Inputs

For modal radiation runs, the structural dry modes of the beam must already exist. Run
the structural/flutter pipeline once for the case before running `run_modal_radiation.py`:

```bash
python main.py ABRAMSON1965
```

This produces the files read by the Capytaine modal mapper:

- `output_data/<case>/<case>_dry_egv_eigendata.csv`
- `output_data/<case>/<case>_dry_egv_nodes.csv`
- `output_data/<case>/<case>_dry_egv_constrained_dofs.csv`

The modal mapper uses these files to reconstruct each beam eigenvector, interpolate it
onto the Capytaine mesh face centers, and register one custom Capytaine DOF per mode.
Rigid-body radiation runs (`--rigid-body` or `rigid_body_motion=True`) do not use these
modal CSV files, but the modal workflow does.

## `build_capytaine_mesh.py`

`build_capytaine_mesh.py` creates the hydrodynamic surface mesh for supported hydrofoil
cases such as `NACA0003` and `ABRAMSON1965`.

The script:

1. Loads the case configuration with `examples/config.py`.
2. Reads the closed airfoil section from `cfg.raw`.
3. Resamples the section if `mesh_n_chord` or `--n-chord` is provided.
4. Scales the section by `cfg.chord` and shifts the elastic-axis position by
   `cfg.xea_factor * chord`.
5. Extrudes the section along the beam span with Gmsh.
6. Loads the generated surface mesh into Capytaine.
7. Applies the same geometric rotations used by the structural/aerodynamic model:
   pitch about `+X`, angle of attack about `+Y`, and dihedral about `+Y`.
8. Exports the mesh, by default to
   `FLUID/capytaine/<case>/<case>_mesh.vtu`.

Typical usage from the repository root:

```bash
python FLUID/capytaine/build_capytaine_mesh.py --case_name ABRAMSON1965
```

Useful options:

- `--n-span`: override spanwise mesh segments.
- `--n-chord`: override the number of perimeter points used on the section.
- `--gmsh-lc`: set the Gmsh characteristic length manually.
- `--pitch`: override `config.pitch`.
- `--offset-z`: translate the hull in `Z` after meshing. The default is zero so the
  hull remains aligned with the beam frame.
- `--use-config-offset-z`: use `config.offset_z` instead of `--offset-z`.
- `--out`: write the mesh to a custom `.vtu`, `.stl`, or `.npz` path.
- `--show` / `--matplotlib`: preview the generated mesh.

## `run_modal_radiation.py`

`run_modal_radiation.py` runs Capytaine radiation problems on either modal beam DOFs or
six rigid-body DOFs.

For the modal workflow, the default command is:

```bash
python FLUID/capytaine/run_modal_radiation.py --case-name ABRAMSON1965
```

With only `--case-name`, the script reads defaults from the case config:

- `mesh_path`: Capytaine mesh to load.
- `modal_dir`: directory containing the dry structural mode CSV files.
- `prefix`: dry mode filename prefix, for example `ABRAMSON1965_dry_egv`.
- `omega_list`: radiation frequencies in rad/s.
- `depth`: one depth or a list of depths used to translate the mesh below the free surface.
- `water_depth` and `free_surface_elevation`: Capytaine water-domain settings.
- `capytaine_results_dir`: output directory for `.npz`, `.csv`, and `.png` files.

At runtime, the script:

1. Loads the Capytaine mesh and translates it by each configured depth.
2. Reconstructs the requested dry beam modes from the CSV files, unless `--modes` is
   supplied.
3. Maps beam translations and rotations to mesh face-center displacements using rigid
   cross-section kinematics.
4. Registers these face displacement fields as custom Capytaine DOFs.
5. Solves one radiation problem for every `(omega, mode)` pair.
6. Assembles added mass and radiation damping matrices in modal space.
7. Saves per-depth results and diagnostic plots.

The main outputs are written in `capytaine_results_dir`, normally
`FLUID/capytaine/<case>/results_modal_radiation/`:

- `modal_radiation_AB_depth_<depth>.npz`: stable workflow file with `omega`,
  `mode_names`, `added_mass`, and `added_damping`.
- `added_mass_depth_<depth>.csv`
- `radiation_damping_depth_<depth>.csv`
- `added_mass_vs_omega_depth_<depth>.png`
- `radiation_damping_vs_omega_depth_<depth>.png`
- `debug_mesh_free_surface_depth_<depth>.png`

For a single-depth run, aggregate compatibility files are also written without the depth
suffix, for example `modal_radiation_AB.npz`, `added_mass.csv`, and
`radiation_damping.csv`.

Useful options:

- `--mode-indices 0,1,2`: choose specific zero-based rows from `*_eigendata.csv`.
- `--modes path/to/file.npz`: bypass beam CSV mapping and load precomputed
  `face_displacements` directly.
- `--rigid-body`: use Capytaine's six rigid-body DOFs instead of modal DOFs.
- `--depth`: override the config depth list with a single value.
- `--omega-list`: override `cfg.omega_list`.
- `--raw-eigenvector`: use the raw mass-normalized eigenvectors instead of normalized
  mode shapes.
- `--plot-mode-projection-debug`: save one plot per mode showing the mapped face
  displacement field on the mesh.
- `--no-plots`: skip matrix sweep plots.

## Connection With Flutter Runs

The flutter solver loads Capytaine matrices from `config.capytaine_results_dir` and the
selected `config.depth_index`. For example, if `config.depth = [0.0508, 0.052]` and
`config.depth_index = 0`, the solver expects:

```text
FLUID/capytaine/<case>/results_modal_radiation/modal_radiation_AB_depth_0p0508.npz
```

After generating or refreshing Capytaine results, run the case normally:

```bash
python main.py ABRAMSON1965
```

## Validation Status

The Capytaine BEM implementation is still under development and should be treated as a
work-in-progress validation path. Current comparisons and validation notes are collected
in the Abramson benchmark report:

- [`docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf`](../../docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf)
- [`docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.tex`](../../docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.tex)
