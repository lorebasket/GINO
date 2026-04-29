# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 18:06 2025

@author: Lorenzo Valle
"""
import os
import time
import platform


# Configure matplotlib to not use LaTeX (to avoid LaTeX missing error)
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

# Import required libraries
import dolfin as do
from dolfin import cpp
from anba4.anbax import anbax
from SONATA.classBlade import Blade
from SONATA.utl.beam_struct_eval import beam_struct_eval

# ==============
# Main
# ==============

start_time = time.time()

print('Current working directory is:', os.getcwd())


# ===== Provide Path Directory & Yaml Filename ===== #
run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
job_str = 'NACA0015.yaml'  # note: for better meshing convergence, units specified in yaml are in 'mm' instead of 'm'
job_name = 'NACA0015'

filename_str = run_dir + job_str


# ===== Define flags ===== #
flag_3d                 = False
flag_wt_ontology        = False
flag_ref_axes_wt        = False

attribute_str           = 'MatID'

flag_plotTheta11        = False     # plane orientation angle
flag_recovery           = False
flag_plotDisplacement   = False     # Needs recovery flag to be activated - shows displacements from loadings in cross sectional plots

flag_wf                 = True      # plot wire-frame
flag_lft                = True      # plot lofted shape of blade surface (flag_wf=True obligatory); Note: create loft with grid refinement without too many radial_stations; can also export step file of lofted shape
flag_topo               = True      # plot mesh topology

# Shape of corners
choose_cutoff = 2    # 0 step, 2 round

# create flag dictionary
flags_dict = {"flag_wt_ontology": flag_wt_ontology, "flag_ref_axes_wt": flag_ref_axes_wt,
              "attribute_str": attribute_str,
              "flag_plotDisplacement": flag_plotDisplacement, "flag_plotTheta11": flag_plotTheta11,
              "flag_wf": flag_wf, "flag_lft": flag_lft, "flag_topo": flag_topo,
              "flag_recovery": flag_recovery}

radial_stations = [0.0, 0.5, 1.0]

# Create job structure
job = Blade(name=job_name, filename=filename_str, flags=flags_dict, stations=radial_stations)
job.blade_gen_section(topo_flag=True, mesh_flag=True)

# Generate topology for each section to ensure points_dict is populated
for section in job.sections:
    section[1].cbm_gen_topo()

# ==== ANBAX Verification ==== #
# vabs_path = "/Users/rfeil/work/8_VABS/vabs_WIN/AnalySwift/VABS/VABSIII.exe"
# job.blade_run_vabs(vabs_path)

flag_DeamDyn_def_transform            = False
flag_write_BeamDyn                    = False 
flag_write_BeamDyn_unit_convert       = ''

flag_3d                               = False
flag_run_vabs                         = True
flag_run_anbax                        = True
flag_verify_vabs_anbax                = False
flag_plot_vabs_struct_characteristics = False
flag_csv_export                       = True  # write BeamDyn input files for follow-up OpenFAST analysis (requires flag_DeamDyn_def_transform = True)

# Update flags dictionary
flags_dict ['flag_run_vabs'] = flag_run_vabs
flags_dict ['flag_run_anbax'] = flag_run_anbax
flags_dict ['flag_verify_vabs_anbax'] = flag_verify_vabs_anbax
flags_dict ['flag_DeamDyn_def_transform'] = flag_DeamDyn_def_transform
flags_dict ['flag_plot_vabs_struct_characteristics'] = flag_plot_vabs_struct_characteristics
flags_dict ['flag_csv_export'] = flag_csv_export
flags_dict ['flag_write_BeamDyn'] = flag_write_BeamDyn
flags_dict ['flag_write_BeamDyn_unit_convert'] = flag_write_BeamDyn_unit_convert
Loads_dict = {
	"Forces": [0., 0., 0.],  # [Fx, Fy, Fz]
	"Moments": [0., 0., 0.]   # [Mx, My, Mz]
}

beam_struct_eval(flags_dict=flags_dict, loads_dict=Loads_dict, cs_pos=radial_stations, job=job, folder_str=run_dir, job_str=job_str, mu=None)

# ===== PLOTS ===== #
#job.blade_plot_attributes()
job.blade_plot_beam_props()

job.blade_plot_sections(attribute=attribute_str, plotTheta11=flag_plotTheta11, plotDisplacement=flag_plotDisplacement, savepath=run_dir)
if flag_3d:
    job.blade_post_3dtopo(flag_wf=flags_dict['flag_wf'], flag_lft=flags_dict['flag_lft'], flag_topo=flags_dict['flag_topo'])

# === EXPORT CBM SECTIONS SUMMARY === #
import os
import csv
import numpy as np

# Create export directory and paths
csv_export_dir = os.path.join(run_dir, 'csv_export')
os.makedirs(csv_export_dir, exist_ok=True)
section_csv_path = os.path.join(csv_export_dir, f"{job_name}_section_data.csv")
contour_csv_path = os.path.join(csv_export_dir, f"{job_name}_contour_points.csv")

# First create section data
section_data = {}
for i, (eta, cbm) in enumerate(job.sections):
    p = cbm.Ax2.Location()
    x1 = cbm.Ax2.XDirection().XYZ()
    x2 = cbm.Ax2.YDirection().XYZ()
    x3 = cbm.Ax2.Direction().XYZ()

    if cbm.BeamProperties:
        sc = cbm.BeamProperties.Xs
        cg = cbm.BeamProperties.Xm
        na = cbm.BeamProperties.Xt
    else:
        raise ValueError("BeamProperties not available. Please run VABS analysis first.")

    section_data[eta] = {
        "position": np.array([p.X(), p.Y(), p.Z()]),
        "frame": {
            "x1": np.array([x1.X(), x1.Y(), x1.Z()]),
            "x2": np.array([x2.X(), x2.Y(), x2.Z()]),
            "x3": np.array([x3.X(), x3.Y(), x3.Z()])
        },
        "shear_center": sc,
        "gravity_center": cg,
        "natural_axes": na
    }

# Write section data CSV
with open(section_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "eta", "X", "Y", "Z",
        "x1_x", "x1_y", "x1_z",
        "x2_x", "x2_y", "x2_z",
        "x3_x", "x3_y", "x3_z",
        "SC_X", "SC_Y",
        "CG_X", "CG_Y",
        "NA_X", "NA_Y"
    ])
    for eta, data in section_data.items():
        pos = data["position"]
        f = data["frame"]
        sc = data["shear_center"]
        cg = data["gravity_center"]
        na = data["natural_axes"]

        writer.writerow([
            eta,
            pos[0], pos[1], pos[2],
            *f["x1"], *f["x2"], *f["x3"],
            *sc, *cg, *na
        ])

print("\n=== Export Complete ===")

# Ensure export directory exists
os.makedirs(csv_export_dir, exist_ok=True)