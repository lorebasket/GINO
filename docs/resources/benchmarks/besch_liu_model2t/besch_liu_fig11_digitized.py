#!/usr/bin/env python3
"""
Digitized benchmark data from:
  Besch, P. K. and Liu, Y-N., "Bending Flutter and Torsional Flutter of
  Flexible Hydrofoil Struts," NSRDC Report 4012, February 1973.
  Figure 11 (page 39): Hydroelastic Mode Characteristics for Model 2T.

Hand-digitized from the scanned figure -> limited precision (+/- ~0.3 Hz,
+/- ~0.005 in damping ratio; speeds +/- ~0.5 kn). Use for qualitative overlay.

Model 2T = blunt-based steel strut (Model 2) + slender tip pod (config G).
  - Flutter (experiment, Ref. 4):           18.4 kn
  - Flutter (NSRDC, mode 2):                 18.1 kn @ 6.4 Hz
  - Flutter (theory, 2-D strip):             14.1 kn  (conservative -22%)
  - Flutter (theory, 3-D strip):             18.8 kn  (slightly nonconservative)
  - Unstable mode = mode 2, predominantly FIRST TORSION.
"""

import numpy as np

# Reference flutter speeds (vertical markers / scalars)
FLUTTER = {
    "experiment_ref4_kn": 18.4,
    "nsrdc_mode2_kn": 18.1,
    "nsrdc_mode2_freq_hz": 6.4,
    "theory_2d_kn": 14.1,
    "theory_3d_kn": 18.8,
}

# ---------------------------------------------------------------------------
# FIGURE 11b  -  FREQUENCY (Hz) vs SPEED (knots), 0-30 kn
# ---------------------------------------------------------------------------
# Mode 1 (predominantly 1st bending, 1B)
FIG11B_MODE1 = np.array([
    [0, 1.9], [5, 2.0], [10, 2.0], [15, 2.1], [18, 2.2],
    [20, 2.4], [25, 3.0], [30, 3.7],
])

# Mode 2 (predominantly 1st torsion, 1T -> the flutter mode) - 3-D theory
FIG11B_MODE2 = np.array([
    [0, 7.9], [5, 7.7], [10, 7.3], [13, 7.0], [15, 6.8],
    [18, 6.5], [20, 6.3], [22, 6.2], [25, 6.1], [30, 6.0],
])

# Mode 3 (predominantly 2nd bending, 2B)
FIG11B_MODE3 = np.array([
    [0, 13.0], [5, 13.1], [10, 13.3], [15, 13.5], [18, 13.4],
    [20, 12.8], [22, 12.2], [25, 12.6], [30, 13.2],
])

# NSRDC measured mode-2 frequency points (open circles along the 1T branch)
FIG11B_EXP_MODE2 = np.array([
    [5, 7.7], [10, 7.2], [12, 7.0], [14, 6.8], [16, 6.6], [18, 6.5],
])

# Flutter frequency points (filled = Ref. 4 / NSRDC at flutter)
FIG11B_FLUTTER_PTS = np.array([
    [18.1, 6.4], [18.4, 6.4],
])

# ---------------------------------------------------------------------------
# FIGURE 11a  -  DAMPING RATIO zeta vs SPEED (knots), 0-30 kn
# ---------------------------------------------------------------------------
# Mode 2 flutter branch, 3-D theory (solid) - crosses zero at 18.8 kn
FIG11A_MODE2_3D = np.array([
    [0, 0.004], [2, 0.010], [5, 0.018], [8, 0.025], [11, 0.028],
    [13, 0.028], [15, 0.024], [16, 0.019], [17, 0.012], [18.8, 0.0],
    [20, -0.010], [22, -0.021], [25, -0.035], [30, -0.052],
])

# Mode 2 flutter branch, 2-D theory (dashed) - crosses zero at 14.1 kn
FIG11A_MODE2_2D = np.array([
    [0, 0.004], [2, 0.010], [5, 0.017], [8, 0.022], [11, 0.020],
    [13, 0.010], [14.1, 0.0], [16, -0.018], [18, -0.035], [20, -0.052],
])

# Mode 3 damping (rises then ~flat)
FIG11A_MODE3 = np.array([
    [0, 0.0], [2, 0.018], [5, 0.038], [8, 0.048], [10, 0.050],
    [13, 0.046], [15, 0.040], [20, 0.030], [25, 0.022], [30, 0.018],
])

# Mode 1 damping (steep rise - heavily damped bending; runs off top of plot)
FIG11A_MODE1 = np.array([
    [0, 0.0], [1, 0.025], [2, 0.045], [3, 0.062], [5, 0.085], [7, 0.10],
])

# NSRDC measured mode-2 damping (open circles, with scatter / error bars)
FIG11A_EXP_MODE2 = np.array([
    [3, 0.020], [5, 0.026], [8, 0.045], [10, 0.052], [12, 0.060],
    [13, 0.055], [15, 0.048], [16, 0.038], [17, 0.028], [18, 0.012],
])


def save_csvs(outdir="."):
    """Write each digitized curve to a CSV file."""
    import os
    sets = {
        "fig11b_mode1_freq": FIG11B_MODE1,
        "fig11b_mode2_freq": FIG11B_MODE2,
        "fig11b_mode3_freq": FIG11B_MODE3,
        "fig11b_exp_mode2_freq": FIG11B_EXP_MODE2,
        "fig11b_flutter_pts": FIG11B_FLUTTER_PTS,
        "fig11a_mode2_3d_damp": FIG11A_MODE2_3D,
        "fig11a_mode2_2d_damp": FIG11A_MODE2_2D,
        "fig11a_mode3_damp": FIG11A_MODE3,
        "fig11a_mode1_damp": FIG11A_MODE1,
        "fig11a_exp_mode2_damp": FIG11A_EXP_MODE2,
    }
    for name, arr in sets.items():
        path = os.path.join(outdir, f"{name}.csv")
        header = "speed_kn,freq_hz" if "freq" in name or name.endswith("pts") else "speed_kn,zeta"
        np.savetxt(path, arr, delimiter=",", header=header, comments="")
    print(f"Wrote {len(sets)} digitized CSV files to {outdir}")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    save_csvs(here)
