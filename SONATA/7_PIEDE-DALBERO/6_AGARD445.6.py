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
job_str = '6_AGARD445.6.yaml'  # note: for better meshing convergence, units specified in yaml are in 'mm' instead of 'm'
job_name = '6_AGARD445.6'
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
flag_recovery           = False
flag_plotDisplacement   = False     # Needs recovery flag to be activated - shows displacements from loadings in cross sectional plots

# 3D plots (blade_post_3dtopo)
flag_wf                 = True      # plot wire-frame
flag_lft                = True      # plot lofted shape of blade surface (flag_wf=True obligatory); Note: create loft with grid refinement without too many radial_stations; can also export step file of lofted shape
flag_topo               = True      # plot mesh topology
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
radial_stations = [0.0, 0.5, 1.0]

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
flag_3d_matplotlib = True  # new: if True, use Matplotlib fallback instead of OCC viewer
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
job.blade_plot_sections(attribute=attribute_str, plotTheta11=flag_plotTheta11, plotDisplacement=flag_plotDisplacement) #, savepath=folder_str)
if flag_3d:
    if flag_3d_matplotlib:
        #job.plot_blade_matplotlib(
        #    savepath=os.path.join(run_dir, f"{job_str}_matplotlib3d.png"),
        #    n_sections=41
        #)
        job.plot_blade_matplotlib(n_sections=41, z_exaggeration=1.0)
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

### === wing geometrical properties === ###
# === GEOMETRIC PROPS (A, Ix, Iy, J) ALONG SPAN ===
import os, csv
import numpy as np
import matplotlib.pyplot as plt

geom_csv_path = os.path.join(csv_export_dir, f"{job_name}_geom_props.csv")

def _cell_xy_vertices(cell):
    """
    Try a few common CBM cell layouts to get a (N,2) array of vertices in local CBM Ax2 frame.
    Adjust these accessors if your CBM differs.
    """
    # 1) Typical: cell has 'nodes' list/array, each node has X(), Y() or .x,.y
    if hasattr(cell, "nodes") and len(cell.nodes) > 0:
        pts = []
        for n in cell.nodes:
            if hasattr(n, "X") and hasattr(n, "Y"):
                pts.append([float(n.X()), float(n.Y())])
            elif hasattr(n, "x") and hasattr(n, "y"):
                pts.append([float(n.x), float(n.y)])
            elif hasattr(n, "Coord"):  # e.g., OCC-like
                xy = n.Coord()
                pts.append([float(xy[0]), float(xy[1])])
        if pts:
            return np.asarray(pts, dtype=float)

    # 2) Some CBM meshes keep coords directly on cell
    for attr in ("pnts2d", "points2d", "pnts", "points", "coords2d", "coords", "coordinates"):
        if hasattr(cell, attr):
            arr = np.asarray(getattr(cell, attr), dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]

    # 3) Fallback: None (we’ll handle gracefully)
    return None

def _poly_area_centroid_Ixy(x, y):
    """
    Shoelace formulas for polygon area A, centroid (cx, cy), and
    second moments about the ORIGIN (Ix0, Iy0, Ixy0).
    Assumes non-self-intersecting polygon; works for CCW/CW (A can be negative; we use abs).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    # close the polygon
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]; y = np.r_[y, y[0]]

    dx = x[:-1]*y[1:] - x[1:]*y[:-1]
    A = 0.5*np.sum(dx)
    Aabs = abs(A)
    if Aabs < 1e-18:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    cx = (1.0/(6.0*A)) * np.sum((x[:-1] + x[1:]) * dx)
    cy = (1.0/(6.0*A)) * np.sum((y[:-1] + y[1:]) * dx)

    Ix0 = (1.0/12.0) * np.sum((y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2) * dx)
    Iy0 = (1.0/12.0) * np.sum((x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2) * dx)
    Ixy0 = (1.0/24.0) * np.sum((x[:-1]*y[1:] + 2*x[:-1]*y[:-1] + 2*x[1:]*y[1:] + x[1:]*y[:-1]) * dx)

    # Return positive area and the corresponding moments (sign of A does not affect magnitudes)
    s = np.sign(A) if A != 0 else 1.0
    return Aabs, cx, cy, s*Ix0, s*Iy0, s*Ixy0

# Collect along span
etas = []
A_list = []
Ix_cg_list = []
Iy_cg_list = []
J_list = []

for (eta, cs) in job.sections:
    etas.append(float(eta))

    # Per-section accumulation
    A_sec = 0.0
    Ax_cx = 0.0
    Ay_cy = 0.0
    # origin-based second moments for parallel-axis shift later
    Ix0_sum = 0.0
    Iy0_sum = 0.0

    # First pass: compute per-cell area and centroid and Ix0, Iy0; build section centroid
    cell_records = []
    for c in cs.mesh:
        # Prefer CBM’s own area if available
        try:
            A_cell = float(c.calc_area())
        except Exception:
            A_cell = None

        # Get vertices and compute (as needed)
        xy = _cell_xy_vertices(c)
        if xy is None:
            # If vertices are unavailable, skip (area too small or unknown)
            continue

        x, y = xy[:,0], xy[:,1]
        A_poly, cx, cy, Ix0, Iy0, Ixy0 = _poly_area_centroid_Ixy(x, y)

        # If CBM area exists, trust it for better consistency
        if A_cell is not None and A_cell > 0:
            A_use = A_cell
        else:
            A_use = A_poly

        if A_use <= 0:
            continue

        cell_records.append((A_use, cx, cy, Ix0, Iy0))
        A_sec += A_use
        Ax_cx += A_use * cx
        Ay_cy += A_use * cy
        Ix0_sum += Ix0
        Iy0_sum += Iy0

    if A_sec <= 0:
        # Degenerate section
        A_list.append(0.0); Ix_cg_list.append(0.0); Iy_cg_list.append(0.0); J_list.append(0.0)
        continue

    # Section centroid
    xbar = Ax_cx / A_sec
    ybar = Ay_cy / A_sec

    # Second pass: shift to centroid using parallel-axis theorem
    Ix_cg = 0.0
    Iy_cg = 0.0
    for (A_use, cx, cy, Ix0, Iy0) in cell_records:
        # Ix about centroid = Ix0 - A*(ybar^2) + A*(cy^2) - 2*A*cy*ybar + A*ybar^2 ???  -> Safer: use direct parallel-axis:
        # Ix_cg_total = sum( Ix_cell_about_its_centroid + A*(dy)^2 )
        # But we have Ix0 about origin, not about cell centroid. Convert:
        # Ix0 = Ix_c + A*cy^2  => Ix_c = Ix0 - A*cy^2
        # Then shift from cell centroid to section centroid: Ix_cg += Ix_c + A*(cy - ybar)^2
        Ix_c = Ix0 - A_use*(cy**2)
        Iy_c = Iy0 - A_use*(cx**2)
        Ix_cg += Ix_c + A_use*(cy - ybar)**2
        Iy_cg += Iy_c + A_use*(cx - xbar)**2

    J = Ix_cg + Iy_cg

    A_list.append(A_sec)
    Ix_cg_list.append(Ix_cg)
    Iy_cg_list.append(Iy_cg)
    J_list.append(J)

# Write CSV
with open(geom_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["eta", "Area_m2", "Ix_m4", "Iy_m4", "J_m4"])
    for eta, A, Ix, Iy, J in zip(etas, A_list, Ix_cg_list, Iy_cg_list, J_list):
        w.writerow([eta, A, Ix, Iy, J])

print(f"[geom] Wrote geometric properties -> {geom_csv_path}")

# Quick plots
etas_np = np.asarray(etas, float)
plt.figure(); plt.plot(etas_np, A_list); plt.xlabel("r/R"); plt.ylabel("Area A [m²]"); plt.title("Area vs span"); plt.grid(True)
plt.figure(); plt.plot(etas_np, Ix_cg_list, label="Ix"); plt.plot(etas_np, Iy_cg_list, label="Iy"); plt.xlabel("r/R"); plt.ylabel("Second moment [m⁴]"); plt.title("I vs span"); plt.legend(); plt.grid(True)
plt.figure(); plt.plot(etas_np, J_list); plt.xlabel("r/R"); plt.ylabel("Polar moment J [m⁴]"); plt.title("J vs span"); plt.grid(True)
plt.show()