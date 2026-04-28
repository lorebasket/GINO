import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# -*- coding: utf-8 -*-
"""
Created on Mo Oct 07 11:18:28 2019

@author: Roland Feil
"""

import os
import numpy as np
import time

from SONATA.classBlade import Blade
from SONATA.utl.beam_struct_eval import beam_struct_eval


# ==============
# Main
# ==============

start_time = time.time()

print('Current working directory is:', os.getcwd())


# ===== Provide Path Directory & Yaml Filename ===== #
run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
job_str = 'wing01_test.yaml'  # note: for better meshing convergence, units specified in yaml are in 'mm' instead of 'm'
job_name = 'wing01_test'
filename_str = run_dir + job_str


# ===== Define flags ===== #
flag_wt_ontology        = False # if true, use ontology definition of wind turbines for yaml files
flag_ref_axes_wt        = False # if true, rotate reference axes from wind definition to comply with SONATA (rotorcraft # definition)

# --- plotting flags ---
# Define mesh resolution, i.e. the number of points along the profile that is used for out-to-inboard meshing of a 2D blade cross section
mesh_resolution = 500

attribute_str           = 'MatID' # default: 'MatID' (theta_3 - fiber orientation angle)
                                            # others:  'theta_3' - fiber orientation angle
                                            #          'stress.sigma11' (use sigma_ij to address specific component)
                                            #          'stressM.sigma11'
                                            #          'strain.epsilon11' (use epsilon_ij to address specific component)
                                            #          'strainM.epsilon11'

# 2D cross sectional plots (blade_plot_sections)
flag_plotTheta11        = False     # plane orientation angle
flag_recovery           = True
flag_plotDisplacement   = True     # Needs recovery flag to be activated - shows displacements from loadings in cross sectional plots

# 3D plots (blade_post_3dtopo)
flag_wf                 = True      # plot wire-frame
flag_lft                = True      # plot lofted shape of blade surface (flag_wf=True obligatory); Note: create loft with grid refinement without too many radial_stations; can also export step file of lofted shape
flag_topo               = False      # plot mesh topology
c2_axis                 = False
flag_DeamDyn_def_transform = True               # transform from SONATA to BeamDyn coordinate system
flag_write_BeamDyn = True                       # write BeamDyn input files for follow-up OpenFAST analysis (requires flag_DeamDyn_def_transform = True)
flag_write_BeamDyn_unit_convert = ''  #'mm_to_m'     # applied only when exported to BeamDyn files

# Shape of corners
choose_cutoff = 2    # 0 step, 2 round

# create flag dictionary
flags_dict = {"flag_wt_ontology": flag_wt_ontology, "flag_ref_axes_wt": flag_ref_axes_wt,
              "attribute_str": attribute_str,
              "flag_plotDisplacement": flag_plotDisplacement, "flag_plotTheta11": flag_plotTheta11,
              "flag_wf": flag_wf, "flag_lft": flag_lft, "flag_topo": flag_topo, "mesh_resolution": mesh_resolution,
              "flag_recovery": flag_recovery,  "c2_axis": c2_axis}

# Define the radial stations for cross sectional analysis (only used for flag_wt_ontology = True -> otherwise, sections from yaml file are used!)
radial_stations = [0.0, 0.059, 0.118, 0.177, 0.236, 0.294, 0.353, 0.412, 0.47, 0.529, 0.588, 0.647, 0.706, 0.765, 0.823, 0.882, 0.941, 1.0]

# ===== Execute SONATA Blade Component Object ===== #
# name          - job name of current task
# filename      - string combining the defined folder directory and the job name
# flags         - communicates flag dictionary (defined above)
# stations      - input of radial stations for cross sectional analysis
# stations_sine - input of radial stations for refinement (only and automatically applied when lofing flag flag_lft = True)
job = Blade(name=job_name, filename=filename_str, flags=flags_dict, stations=radial_stations)

# ===== Build & mesh segments ===== #
job.blade_gen_section(topo_flag=True, mesh_flag = True)

# Shape of corners
choose_cutoff = 2    # 0 step, 2 round
# ===== Recovery Analysis + BeamDyn Outputs ===== #

job.blade_run_anbax()

# Define flags
flag_3d = True
flag_3d_matplotlib = True  # if True use Matplotlib fallback; set False to use native OCC viewer and blade_post_3dtopo
flag_csv_export = True                         # export csv files with structural data
# Update flags dictionary
flags_dict['flag_csv_export'] = flag_csv_export
flags_dict['flag_DeamDyn_def_transform'] = flag_DeamDyn_def_transform
flags_dict['flag_write_BeamDyn'] = flag_write_BeamDyn
flags_dict['flag_write_BeamDyn_unit_convert'] = flag_write_BeamDyn_unit_convert
Loads_dict = {"Forces":[1.,1.,1.],"Moments":[1.,1.,1.]}

# Set damping for BeamDyn input file
delta = np.array([0.03, 0.03, 0.06787]) # logarithmic decrement, natural log of the ratio of the amplitudes of any two successive peaks. 3% flap and edge, 6% torsion
zeta = 1. / np.sqrt(1.+(2.*np.pi / delta)**2.) # damping ratio,  dimensionless measure describing how oscillations in a system decay after a disturbance
omega = np.array([0.508286, 0.694685, 4.084712])*2*np.pi # Frequency (rad/s), flap/edge/torsion
mu1 = 2*zeta[0]/omega[0]
mu2 = 2*zeta[1]/omega[1]
mu3 = 2*zeta[2]/omega[2]
mu = np.array([mu1, mu2, mu3, mu2, mu1, mu3])
beam_struct_eval(flags_dict, Loads_dict, radial_stations, job, run_dir, job_str, mu)


# ===== PLOTS ===== #
# Create plots directory
plots_dir = os.path.join(run_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Save cross-section plots
job.blade_plot_sections(attribute=attribute_str, plotTheta11=flag_plotTheta11, plotDisplacement=flag_plotDisplacement, savepath=plots_dir)

# Save 3D plot
if flag_3d:
    if flag_3d_matplotlib:
        job.plot_blade_matplotlib(
            savepath=os.path.join(plots_dir, f"{job_name}_3d_blade.png"),
            use_sections=True, 
            z_exaggeration=1.0,
            interactive=True,  # Show AND save
            figsize=(16, 12),  # Larger figure for better clarity
            dpi=300            # High resolution for saved image
        )
        # Also save the multi-view image (perspective/top/side/front) to plots
        try:
            mv_path = os.path.join(plots_dir, f"{job_name}_multiview.png")
            job.plot_blade_matplotlib(
                savepath=mv_path,
                use_sections=True,
                z_exaggeration=1.0,
                interactive=False,  # do not show second window
                figsize=(16, 12),
                dpi=300
            )
            print(f"[matplotlib] Saved multi-view plot -> {mv_path}")
        except Exception as e:
            print(f"[matplotlib] Warning: could not save multi-view plot: {e}")
    else:
        job.blade_post_3dtopo(
            flag_wf=flags_dict['flag_wf'],
            flag_lft=flags_dict['flag_lft'],
            flag_topo=flags_dict['flag_topo']
        )

print("--- Computational time: %s seconds ---" % (time.time() - start_time))

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
        "SC_global_X", "SC_global_Y", "SC_global_Z",
        "CG_X", "CG_Y",
        "CG_global_X", "CG_global_Y", "CG_global_Z",
        "NA_X", "NA_Y"
    ])
    for eta, data in section_data.items():
        pos = data["position"]
        f = data["frame"]
        sc = data["shear_center"]
        cg = data["gravity_center"]
        na = data["natural_axes"]
        
        # Calculate global coordinates for shear center
        # VABS convention: sc[0] is negative offset along x2 direction
        # sc_global = pos - sc[0] * x2 + sc[1] * x3
        sc_global = pos - sc[0] * f["x2"] + sc[1] * f["x3"]
        
        # Calculate global coordinates for center of gravity
        # Same VABS convention: cg[0] is negative offset along x2 direction
        cg_global = pos - cg[0] * f["x2"] + cg[1] * f["x3"]

        writer.writerow([
            eta,
            pos[0], pos[1], pos[2],
            *f["x1"], *f["x2"], *f["x3"],
            *sc,
            sc_global[0], sc_global[1], sc_global[2],
            *cg,
            cg_global[0], cg_global[1], cg_global[2],
            *na
        ])

print("\n=== Export Complete ===")
print(f"Section data (with global coordinates) saved to: {section_csv_path}")