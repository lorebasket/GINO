# agard_area.py
import numpy as np
import yaml

def _polygon_area(x, y):
    # Gauss's area formula
    x_shift = np.roll(x, -1)
    y_shift = np.roll(y, -1)
    A = 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))
    return A

def _polygon_perimeter(x, y):
    dx = np.diff(np.r_[x, x[0]])
    dy = np.diff(np.r_[y, y[0]])
    perimeter = float(np.sum(np.hypot(dx, dy)))

    return perimeter

def load_geometry(yaml_path, airfoil_name):
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f)
    af = [a for a in yml["airfoils"] if a["name"] == airfoil_name][0]
    x = np.array(af["coordinates"]["x"], float)
    y = np.array(af["coordinates"]["y"], float)

    chord_grid = np.array(yml["components"]["blade"]["outer_shape_bem"]["chord"]["grid"], float)
    chord_vals = np.array(yml["components"]["blade"]["outer_shape_bem"]["chord"]["values"], float)

    # semispan (m) from reference axis (used elsewhere; not needed for A)
    ref_x = np.array(yml["components"]["blade"]["outer_shape_bem"]["reference_axis"]["x"]["values"], float)
    L_semispan = float(ref_x[-1] - ref_x[0])
    return x, y, chord_grid, chord_vals, L_semispan

def make_A_of_eta(yaml_path, airfoil_name,
                  thin_walled=False, wall_thickness=5e-4):
    """
    Returns:
        A_of_eta(eta) -> area in m^2 at nondimensional span eta ∈ [0,1]
        meta: dict with unit-chord area, perimeter, etc.
    """
    x, y, chord_grid, chord_vals, L_semispan = load_geometry(yaml_path, airfoil_name)
    A_unit = _polygon_area(x, y)            # m^2 (unit chord)
    P_unit = _polygon_perimeter(x, y)       # m   (unit chord)

    def c_of_eta(eta):
        """
        returns the chord length at the given spanwise location(s) eta
        """
        eta_array = np.asarray(eta, dtype=float)
        chord_eta = np.interp(eta_array, chord_grid, chord_vals)
        
        return chord_eta

    if thin_walled:
        # material area of a closed thin-walled contour: A = t * perimeter_scaled
        # perimeter scales linearly with chord
        def A_of_eta(eta):
            c = c_of_eta(eta)
            return wall_thickness * (P_unit * c)
    
    else:
        # solid fill (your YAML has `filler: 2`): area scales with c^2
        def A_of_eta(eta):
            c = c_of_eta(eta)
            return A_unit * (c**2)

    meta = {
        "A_unit_m2": A_unit,
        "P_unit_m": P_unit,
        "chord_grid": chord_grid,
        "chord_vals_m": chord_vals,
        "L_semispan_m": L_semispan,
        "mode": "thin_walled" if thin_walled else "solid_fill",
        "wall_thickness_m": wall_thickness
    }
    return A_of_eta, meta

def sample_A_midpoints(n_el, A_of_eta):
    eta_mid = (np.arange(n_el) + 0.5) / n_el
    A_mid = A_of_eta(eta_mid)
    return eta_mid, A_mid

def unit_inertias(yaml_path, airfoil_name, thin_walled=False, wall_thickness=5e-4):
    import numpy as np, yaml
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f)
    af = [a for a in yml["airfoils"] if a["name"] == airfoil_name][0]
    X = np.column_stack([af["coordinates"]["x"], af["coordinates"]["y"]]).astype(float)

    # polygon area & centroid
    x, y = X[:,0], X[:,1]
    A2 = np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))
    A = 0.5*A2
    
    cx = (1/(6*A)) * np.sum((x + np.roll(x,-1)) * (x*np.roll(y,-1) - y*np.roll(x,-1)))
    cy = (1/(6*A)) * np.sum((y + np.roll(y,-1)) * (x*np.roll(y,-1) - y*np.roll(x,-1)))
    x_c = x - cx; y_c = y - cy

    # Green’s theorem (unit-chord, about centroid)
    Ixx_unit = abs((1/12)*np.sum((y_c**2 + y_c*np.roll(y_c,-1) + np.roll(y_c,-1)**2) *
                                  (x_c*np.roll(y_c,-1) - y_c*np.roll(x_c,-1))))
    Iyy_unit = abs((1/12)*np.sum((x_c**2 + x_c*np.roll(x_c,-1) + np.roll(x_c,-1)**2) *
                                  (x_c*np.roll(y_c,-1) - y_c*np.roll(x_c,-1))))

    # chord distribution
    xg, yg, chord_grid, chord_vals, L_semispan = load_geometry(yaml_path, airfoil_name)

    def c_of_eta(eta):
        eta = np.asarray(eta, dtype=float)
        return np.interp(eta, chord_grid, chord_vals)

    # scaling law: solid => c^4 ; thin-walled (constant thickness) => c^3
    p = 3 if thin_walled else 4

    # Axis mapping: beam Iy is about the vertical (thickness) axis -> Iyy_unit;
    # beam Iz is about the chordwise axis -> Ixx_unit.
    def Iy_of_eta(eta):
        c = c_of_eta(eta)
        return Iyy_unit * (c**p)

    def Iz_of_eta(eta):
        c = c_of_eta(eta)
        return Ixx_unit * (c**p)

    return Iy_of_eta, Iz_of_eta

def calibrate_area_to_total_mass(A_of_eta, beam_length_m, rho, M_target, n=400):
    """
    Returns A_of_eta_cal(eta) = s * A_of_eta(eta) with s set so that
    ∫_0^{beam_length} rho*A(s/L)*ds = M_target.
    """
    import numpy as np

    L = float(beam_length_m)
    s = (np.arange(n) + 0.5) / n # evenly spaced midpoints
    eta_mid = s  # eta btwn [0,1] along the elastic axis
    A_mid = np.asarray(A_of_eta(eta_mid), float)
    M_current = rho * np.trapz(A_mid, s) * L
    
    if M_current <= 0:
        raise ValueError("Non-positive current mass; check A_of_eta.")
    
    scale = float(M_target) / float(M_current)

    def A_of_eta_cal(eta):
        return scale * A_of_eta(eta)

    return A_of_eta_cal, dict(scale=scale, M_current=M_current)
