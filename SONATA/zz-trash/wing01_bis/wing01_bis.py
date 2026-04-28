"""
wing01_bis.py
=============
Runner script for the wing01_bis case study.

This is a translation of wing01.py that uses the **wt-ontology** YAML format
(flag_wt_ontology = True), mirroring the approach adopted by the SONATA
developers in the IEA-15-240-RWT example.

Key differences vs wing01.py
-----------------------------
| Setting               | wing01.py          | wing01_bis.py           |
|-----------------------|--------------------|-------------------------|
| flag_wt_ontology      | False              | **True**                |
| flag_ref_axes_wt      | False              | False (fixed wing ≠ wt) |
| YAML format           | sections-based     | **layers-based**        |
| radial_stations       | defined but unused | **actively used**       |

All other flags (recovery, BeamDyn, CSV export, damping, mesh, plots) are
kept identical to wing01.py so that results can be directly compared.
"""

import os
import sys
import time

start_time = time.time()

# ── Paths ──────────────────────────────────────────────────────────────────── #
# Locate the SONATA package from the git-clone directory relative to this file.
script_dir = os.path.dirname(os.path.abspath(__file__))
sonata_root = os.path.join(script_dir, '..', '9_gitclone')  # adjust if needed
sys.path.insert(0, os.path.abspath(sonata_root))

from SONATA.classBlade import Blade
from SONATA.utl.beam_struct_eval import beam_struct_eval

# ── Job identifiers ───────────────────────────────────────────────────────── #
job_name   = 'wing01_bis'
job_str    = os.path.join(script_dir, 'wing01_bis.yaml')
run_dir    = script_dir

# ── Flags ─────────────────────────────────────────────────────────────────── #
# *** Main change: wt-ontology mode ***
flag_wt_ontology     = True    # ← True  (layers/webs YAML format)
flag_ref_axes_wt     = False   # ← False (wing, not a turbine — no axis rotation)

flag_recovery        = True
flag_write_BeamDyn   = True
flag_csv_export      = True

# Plot flags
flag_3d              = True
flag_3d_matplotlib   = True    # Use matplotlib fallback instead of PythonOCC 3D viewer
flag_plotTheta11     = False
flag_plotDisplacement = False

# ── Mesh & attribute ──────────────────────────────────────────────────────── #
mesh_resolution = 500
attribute_str   = 'MatID'
choose_cutoff   = 2   # round corners

# ── Radial stations ───────────────────────────────────────────────────────── #
# When flag_wt_ontology=True these stations are actively used to pick the
# spanwise locations at which cross-sections are computed.
# Identical to wing01.py for a fair comparison.
radial_stations = [
    0.0, 0.059, 0.118, 0.177, 0.236, 0.294, 0.353,
    0.412, 0.47, 0.529, 0.588, 0.647, 0.706, 0.765,
    0.823, 0.882, 0.941, 1.0
]

# ── flags_dict (passed to post-processing helpers) ────────────────────────── #
flags_dict = {
    'flag_recovery'    : flag_recovery,
    'flag_wf'          : False,
    'flag_lft'         : False,
    'flag_topo'        : False,
    'flag_wt_ontology' : flag_wt_ontology,
    'flag_ref_axes_wt' : flag_ref_axes_wt,
}

# ── Build Blade object ────────────────────────────────────────────────────── #
job = Blade(
    name             = job_name,
    filename         = job_str,
    flag_wt_ontology = flag_wt_ontology,
    flag_ref_axes_wt = flag_ref_axes_wt,
)

# ── Generate cross-sections ───────────────────────────────────────────────── #
job.blade_gen_section(
    stations         = radial_stations,
    mesh_resolution  = mesh_resolution,
    attribute_str    = attribute_str,
    split_quads      = True,
    choose_cutoff    = choose_cutoff,
)

# ── Run VABS / ANBAX structural solver ───────────────────────────────────── #
job.blade_run_anbax()

# ── Beam structural evaluation (damping + BeamDyn) ────────────────────────── #
import numpy as np

# Damping parameters — identical to wing01.py
delta = [0.03, 0.03, 0.06787]
omega = [
    0.508286  * 2 * np.pi,
    0.694685  * 2 * np.pi,
    4.084712  * 2 * np.pi,
]

# Loads for structural recovery (unit loads)
Loads_dict = {
    "Forces" : [1., 1., 1.],
    "Moments": [1., 1., 1.],
}

beam_struct_eval(
    job,
    flag_csv_export    = flag_csv_export,
    flag_write_BeamDyn = flag_write_BeamDyn,
    flag_recovery      = flag_recovery,
    Loads_dict         = Loads_dict,
    damping_delta      = delta,
    damping_omega      = omega,
    run_dir            = run_dir,
    job_name           = job_name,
)

# ── Plots ─────────────────────────────────────────────────────────────────── #
plots_dir = os.path.join(run_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Cross-section plots
job.blade_plot_sections(
    attribute        = attribute_str,
    plotTheta11      = flag_plotTheta11,
    plotDisplacement = flag_plotDisplacement,
    savepath         = plots_dir,
)

# 3-D blade plot
if flag_3d:
    if flag_3d_matplotlib:
        job.plot_blade_matplotlib(
            savepath       = os.path.join(plots_dir, f"{job_name}_3d_blade.png"),
            use_sections   = True,
            z_exaggeration = 1.0,
            interactive    = True,
            figsize        = (16, 12),
            dpi            = 300,
        )
        try:
            mv_path = os.path.join(plots_dir, f"{job_name}_multiview.png")
            job.plot_blade_matplotlib(
                savepath       = mv_path,
                use_sections   = True,
                z_exaggeration = 1.0,
                interactive    = False,
                figsize        = (16, 12),
                dpi            = 300,
            )
            print(f"[matplotlib] Saved multi-view plot -> {mv_path}")
        except Exception as e:
            print(f"[matplotlib] Warning: could not save multi-view plot: {e}")
    else:
        job.blade_post_3dtopo(
            flag_wf   = flags_dict['flag_wf'],
            flag_lft  = flags_dict['flag_lft'],
            flag_topo = flags_dict['flag_topo'],
        )

print("--- Computational time: %s seconds ---" % (time.time() - start_time))

# ── CSV export of cross-section data ─────────────────────────────────────── #
import csv

csv_export_dir  = os.path.join(run_dir, 'csv_export')
os.makedirs(csv_export_dir, exist_ok=True)
section_csv_path = os.path.join(csv_export_dir, f"{job_name}_section_data.csv")

section_data = {}
for i, (eta, cbm) in enumerate(job.sections):
    p  = cbm.Ax2.Location()
    x1 = cbm.Ax2.XDirection().XYZ()
    x2 = cbm.Ax2.YDirection().XYZ()
    x3 = cbm.Ax2.Direction().XYZ()

    if cbm.BeamProperties:
        sc = cbm.BeamProperties.Xs
        cg = cbm.BeamProperties.Xm
        na = cbm.BeamProperties.Xt
    else:
        raise ValueError("BeamProperties not available. Run VABS/ANBAX first.")

    section_data[eta] = {
        "position" : np.array([p.X(), p.Y(), p.Z()]),
        "frame"    : {
            "x1": np.array([x1.X(), x1.Y(), x1.Z()]),
            "x2": np.array([x2.X(), x2.Y(), x2.Z()]),
            "x3": np.array([x3.X(), x3.Y(), x3.Z()]),
        },
        "shear_center"  : sc,
        "gravity_center": cg,
        "natural_axes"  : na,
    }

with open(section_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "eta", "X", "Y", "Z",
        "x1_x", "x1_y", "x1_z",
        "x2_x", "x2_y", "x2_z",
        "x3_x", "x3_y", "x3_z",
        "SC_X", "SC_Y",
        "SC_global_X", "SC_global_Y", "SC_global_Z",
        "CG_X", "CG_Y",
        "CG_global_X", "CG_global_Y", "CG_global_Z",
        "NA_X", "NA_Y",
    ])
    for eta, data in section_data.items():
        pos = data["position"]
        f   = data["frame"]
        sc  = data["shear_center"]
        cg  = data["gravity_center"]
        na  = data["natural_axes"]

        sc_global = pos - sc[0] * f["x2"] + sc[1] * f["x3"]
        cg_global = pos - cg[0] * f["x2"] + cg[1] * f["x3"]

        writer.writerow([
            eta,
            pos[0], pos[1], pos[2],
            *f["x1"], *f["x2"], *f["x3"],
            *sc,
            sc_global[0], sc_global[1], sc_global[2],
            *cg,
            cg_global[0], cg_global[1], cg_global[2],
            *na,
        ])

print("\n=== Export Complete ===")
print(f"Section data saved to: {section_csv_path}")
