# FSI/Hydroelastic_analysis_workflow/structural_model.py

import numpy as np
import sys
import os
import importlib.util as _ilu
from FEA.fea_utl.multibody_assembly import T6_from_beam_direction
from FEA.fea_utl.beam_properties import create_beam_model
from FEA.fea_utl.multibody_assembly import rotate_beam_model_to_global,translate_beam_model


def build(config):
    """
    Builds the structural beam model based on the provided configuration.
    """
    sys.path.extend([
        config.paths['FSI'],
        config.paths['SONATA'],
        config.paths['FEA']
    ])

    if config.name == 'GOLAND':
        return _build_goland_model(config)
    elif config.name == 'GOLANDfullspan':
        return _build_goland_fullspan_model(config)
    elif config.name == 'grid_conv':
        return _build_gridconv_model(config)
    elif config.name == 'wing01':
        return _build_sonata_model(config)
    elif config.name == 'GOLAND_sonata':
        return _build_sonata_model(config)
    elif config.name == 'hollowell':
        return _build_sonata_model(config)
    elif config.name == 'tnz_multibody':
        return _build_tnz_multibody_model(config)
    elif config.name == 'NACA0003':
        return _build_naca0003_model(config)
    elif config.name == 'ABRAMSON1965':
        return _build_abramson_model(config)
    else:
        raise NotImplementedError(f"Structural model build not implemented for {config.name}")


def _build_naca0003_model(config):
    """
    Builds a NACA0003 hydrofoil structural beam model.
 
    Section properties are derived from the normalised airfoil coordinates
    (config.raw) and material properties (config.E, config.rho_s, config.v).
    EIxx is then calibrated to reproduce the known vacuum first natural
    frequency f₁,vac = 321 Hz (Cupr et al. 2018) for the double-clamped
    configuration.
    """
    from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y
    from FEA.fea_utl.beam_properties import create_beam_model
 
    print("Building NACA0003 structural model from airfoil geometry")
 
    # =========================================================================
    # 1.  CONFIGURATION PARAMETERS  (read-only, all come from config.py)
    # =========================================================================
    E           = config.E              # Young's modulus                  [Pa]
    shear_factor = config.shear_factor  # Shear factor                     [–]
    xcm_factor = config.xcm_factor      # CG offset factor                 [–]
    xea_factor = config.xea_factor      # EA offset factor                 [–]
    rho_s       = config.rho_s          # Material density                 [kg/m³]
    v_poisson   = config.v              # Poisson's ratio                  [–]
    beam_length = config.beam_length    # Full span (clamped-clamped)      [m]
    chord       = config.chord          # Chord length                     [m]
    pitch       = config.pitch          # Pitch angle                      [deg]
    raw_coords  = config.raw            # Normalised airfoil coords (N×2)
    n_elements  = config.n_elements     # Number of beam elements
 
    G = E / (2.0 * (1.0 + v_poisson))  # Shear modulus  [Pa]
 
    print(f"  Material : E={E:.3e} Pa  ρ={rho_s} kg/m³  ν={v_poisson}")
    print(f"  G        = {G:.3e} Pa")
    print(f"  Geometry : L={beam_length:.4f} m  c={chord:.4f} m")
    print(f"  Elements : {n_elements}")
 
    # =========================================================================
    # 2.  CROSS-SECTIONAL GEOMETRY  (all in physical metres)
    # =========================================================================
    x_norm = raw_coords[:, 0]
    y_norm = raw_coords[:, 1]
    x_phys = x_norm * chord
    y_phys = y_norm * chord
    n_pts  = len(x_phys)
 
    # ── 2a.  Area (shoelace – unchanged, exact for polygon) ──────────────────
    area = 0.0
    for i in range(n_pts - 1):
        area += x_phys[i] * y_phys[i + 1] - x_phys[i + 1] * y_phys[i]
    area = 0.5 * abs(area)
 
    # ── 2b.  Centroid (Green's theorem – unchanged) ───────────────────────────
    x_centroid = y_centroid = 0.0
    for i in range(n_pts - 1):
        dA  = 0.5 * (x_phys[i] * y_phys[i + 1] - x_phys[i + 1] * y_phys[i])
        x_c = (x_phys[i] + x_phys[i + 1]) / 3.0
        y_c = (y_phys[i] + y_phys[i + 1]) / 3.0
        x_centroid += x_c * dA
        y_centroid += y_c * dA
    x_centroid = x_centroid / area if area > 0 else 0.0
    y_centroid = y_centroid / area if area > 0 else 0.0
 
    # ── 2c.  Second moments – FIX 1: Green's theorem exact formula ───────────

    Ixx_0 = 0.0
    Iyy_0 = 0.0
    Ixy_0 = 0.0
    for i in range(n_pts - 1):
        xi,  yi  = x_phys[i],     y_phys[i]
        xi1, yi1 = x_phys[i + 1], y_phys[i + 1]
        cross = xi * yi1 - xi1 * yi                    # twice the signed dA
        Ixx_0 += (yi**2  + yi * yi1  + yi1**2)  * cross
        Iyy_0 += (xi**2  + xi * xi1  + xi1**2)  * cross
        Ixy_0 += (xi * yi1 + 2.0 * xi * yi + 2.0 * xi1 * yi1 + xi1 * yi) * cross
 
    Ixx_0 /= 12.0
    Iyy_0 /= 12.0
    Ixy_0 /= 24.0
 
    # Parallel axis theorem → centroidal values
    Ixx = abs(Ixx_0) - area * y_centroid**2
    Iyy = abs(Iyy_0) - area * x_centroid**2
    # Ixy not used in the beam model (symmetric section → Ixy ≈ 0)
 
    # ── 2d.  Torsional constant – FIX 2: Saint-Venant thin-section formula ───

    upper_mask   = y_norm >= 0.0
    x_up_norm    = x_norm[upper_mask]
    y_up_norm    = y_norm[upper_mask]
    sort_idx     = np.argsort(x_up_norm)
    x_up_phys    = x_up_norm[sort_idx] * chord
    y_up_phys    = y_up_norm[sort_idx] * chord
 
    # Integrate (2·y)³ using the trapezoidal rule over the upper surface
    J = (1.0 / 3.0) * float(np.trapz((2.0 * y_up_phys)**3, x_up_phys))
 
    print(f"\n  Cross-sectional properties (Green's theorem / Saint-Venant):")
    print(f"    Area A          = {area:.6e} m²")
    print(f"    Centroid        = ({x_centroid*1e3:.3f} mm, {y_centroid*1e3:.3f} mm)")
    print(f"    Ixx (flapwise)  = {Ixx:.6e} m⁴")
    print(f"    Iyy (edgewise)  = {Iyy:.6e} m⁴")
    print(f"    J_SV (torsion)  = {J:.6e} m⁴  [was Ip={Ixx+Iyy:.3e} m⁴]")
 
    # =========================================================================
    # 3.  SECTIONAL STIFFNESS  (geometric estimates before calibration)
    # =========================================================================
    EA    = E * area
    EIxx  = E * Ixx      # flapwise bending (thin direction) – to be calibrated
    EIyy  = E * Iyy      # edgewise bending (chord direction)
    GJ    = G * J        # torsion – now physically correct
 
    GAx = config.shear_factor * G * area
    GAz = config.shear_factor * G * area
 
    # =========================================================================
    # 4.  MASS PROPERTIES  (solid-section estimate – considered reliable)
    # =========================================================================
    mu  = rho_s * area          # mass per unit length   [kg/m]
    i11 = rho_s * Iyy           # rotational inertia / m about section x
    i22 = rho_s * J             # rotational inertia / m about section y (torsion)
    i33 = rho_s * Ixx           # rotational inertia / m about section z
    #e_x = 0.0
    e_x = (xcm_factor - xea_factor) * chord      # CG offset (symmetric NACA0003 → zero)
 
    print(f"\n  Mass properties:")
    print(f"    μ   = {mu:.6e} kg/m   (total mass = {mu*beam_length:.4f} kg)")
    print(f"    i11 = {i11:.6e} kg·m")
    print(f"    i22 = {i22:.6e} kg·m")
    print(f"    i33 = {i33:.6e} kg·m")
 
    # =========================================================================
    # 5.  EIxx CALIBRATION – FIX 3
    f1_vac_target = 321.0          # Hz  — Cupr et al. 2018
    beta1L        = 4.7300         # dimensionless — clamped-clamped mode 1
    omega1_target = 2.0 * np.pi * f1_vac_target
 
    EIxx_geom = EIxx               # geometry-only estimate (for reporting)
    EIxx      = mu * (omega1_target * beam_length**2 / beta1L**2)**2
    k_calib   = EIxx / EIxx_geom
 
    # Update mass rotational inertia for torsion consistently (J stays geometric)
    # i33 uses Ixx; now replace with calibrated Ixx equivalent for consistency:
    Ixx_cal = EIxx / E
    i33     = rho_s * Ixx_cal     # keep mass rotational inertia consistent
 
    print(f"\n  ── EIxx frequency calibration ─────────────────────────────")
    print(f"    f₁,vac target  = {f1_vac_target:.1f} Hz  (Cupr et al. 2018)")
    print(f"    β₁L            = {beta1L}  (clamped-clamped first mode)")
    print(f"    EIxx geometry  = {EIxx_geom:.4e} N·m²  (solid section)")
    print(f"    EIxx calibrated= {EIxx:.4e} N·m²")
    print(f"    Calibration k  = {k_calib:.4f}  (solid overpredicts by {1/k_calib:.2f}×)")
    print(f"  ────────────────────────────────────────────────────────────")
 
    # Quick analytical check
    f1_check = (beta1L**2) / (2.0 * np.pi * beam_length**2) * np.sqrt(EIxx / mu)
    print(f"    Analytical f₁ check = {f1_check:.2f} Hz  (should be {f1_vac_target:.1f} Hz)")
 
    print(f"\n  Final sectional stiffness:")
    print(f"    EA   = {EA:.4e} N")
    print(f"    EIxx = {EIxx:.4e} N·m²  (calibrated, flapwise)")
    print(f"    EIyy = {EIyy:.4e} N·m²  (geometric,  edgewise)")
    print(f"    GJ   = {GJ:.4e} N·m²  (Saint-Venant torsion)")
    print(f"    GAx  = {GAx:.4e} N")
    print(f"    GAz  = {GAz:.4e} N")
 
    # =========================================================================
    # 6.  PER-ELEMENT K AND M MATRICES
    # =========================================================================
    K_dict = {}
    M_dict = {}
 
    for idx in range(n_elements):
        # ── Stiffness (6×6) ──────────────────────────────────────────────────
        # DOF ordering: [u, v, w, θx, θy, θz]
        #   v  = axial (span, Y)
        #   u  = chordwise (X), w  = thickness (Z)
        #   θx = rotation about X (flapwise bending moment)
        #   θy = rotation about Y (torsion, span axis)
        #   θz = rotation about Z (edgewise bending moment)
        K_i = np.zeros((6, 6), float)
        K_i[0, 0] = GAx    # shear X (chordwise)
        K_i[1, 1] = EA     # axial Y (span)
        K_i[2, 2] = GAz    # shear Z (thickness)
        K_i[3, 3] = EIxx   # flapwise bending  (calibrated)
        K_i[4, 4] = GJ     # torsion           (Saint-Venant)
        K_i[5, 5] = EIyy   # edgewise bending  (geometric)
        K_dict[f"e{idx}"] = K_i
 
        # ── Mass (6×6) ───────────────────────────────────────────────────────
        M_i = np.zeros((6, 6), float)
        M_i[0, 0] = mu
        M_i[1, 1] = mu
        M_i[2, 2] = mu
        # EA-CG offset coupling (zero for symmetric NACA0003)
        M_i[1, 5] = -mu * e_x
        M_i[5, 1] = -mu * e_x
        M_i[2, 4] = +mu * e_x
        M_i[4, 2] = +mu * e_x
        # Rotational inertias at elastic axis (= centroid for symmetric foil)
        M_i[3, 3] = i11
        M_i[4, 4] = i22 + mu * e_x**2
        M_i[5, 5] = i33 + mu * e_x**2
        M_dict[f"e{idx}"] = M_i
 
    # =========================================================================
    # 7.  ASSEMBLE BEAM MODEL
    # =========================================================================
    pitch = getattr(config, 'pitch', 0)
 
    beam_model = create_beam_model(
        K_dict, M_dict,
        beam_length, n_elements, pitch,
        agard_theory=True,
        flutter_benchmark=False,
        sonata=False,
        center_beam=True,
        chord=chord,
    )
 
    alpha_deg = getattr(config, 'alpha_deg', 0.0)
    beam_model = rotate_beam_model_y(beam_model, [alpha_deg])
 
    # Store for downstream diagnostics
    beam_model['K_section'] = K_dict[f"e{n_elements // 2}"]
    beam_model['M_section'] = M_dict[f"e{n_elements // 2}"]
    beam_model['K_dict']    = K_dict
    beam_model['M_dict']    = M_dict
    beam_model['airfoil_coords'] = raw_coords
    beam_model['section_properties'] = {
        'area'          : area,
        'Ixx'           : Ixx_cal,     # calibrated (= EIxx/E)
        'Iyy'           : Iyy,
        'J'             : J,           # Saint-Venant
        'centroid'      : (x_centroid, y_centroid),
        'mu'            : mu,
        'i11'           : i11,
        'i22'           : i22,
        'i33'           : i33,
        'k_EI_calib'    : k_calib,     # calibration factor for diagnostics
        'EIxx_geom'     : EIxx_geom,   # geometric prediction (for reference)
        'f1_vac_target' : f1_vac_target,
    }
 
    # =========================================================================
    # 8.  WEIGHT COMPUTATION
    # =========================================================================
    g = 9.81                                    # Standard gravitational acceleration [m/s²]
    total_mass = mu * beam_length               # Total mass [kg]
    total_weight = total_mass * g               # Total weight [N]
    
    print(f"\nNACA0003 beam built: {n_elements} elements  span={beam_length:.3f} m")
    print(f"  Total structural mass : {total_mass:.4f} kg")
    print(f"  Total structural weight: {total_weight:.4f} N  ({total_weight/g:.4f} kgf)")
 
    beam_model['section_properties']['total_mass'] = total_mass
    beam_model['section_properties']['total_weight'] = total_weight
    beam_model['section_properties']['g'] = g

    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        K = beam_model['K_section']
        M = beam_model['M_section']
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            np.savetxt(k_file, K, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model


def _build_abramson_model(config):
    
    from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y
    from FEA.fea_utl.beam_properties import create_beam_model
    
    print(f"Building ABRAMSON structural model")

    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis
    # u: displacement in X, v: displacement in Y (axial), w: displacement in Z
    # θx: rotation about X, θy: rotation about Y (torsion), θz: rotation about Z
    K = np.zeros((6, 6), float)
    K[0, 0] = config.GAx # Shear stiffness X (transverse)
    K[1, 1] = config.EA  # Axial stiffness along Y (beam axis)
    K[2, 2] = config.GAz # Shear stiffness Z (transverse)
    K[3, 3] = config.EIxx # Bending stiffness about X
    K[4, 4] = config.GJ  # Torsional stiffness (rotation along Y - beam axis)
    K[5, 5] = config.EIzz # Bending stiffness about Z

    xcm = config.xcm_factor * config.chord
    xea = config.xea_factor * config.chord
    e = (xcm - xea)  # Offset in X direction: positive if CG is aft of EA
    #e = 0
    
    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis (spanwise)
    M = np.zeros((6, 6), float)
    M[0, 0] = config.mu  # mass for u (chordwise X)
    M[1, 1] = config.mu  # mass for v (axial Y, span)
    M[2, 2] = config.mu  # mass for w (vertical Z)

    # Off-diagonal coupling from -mu*skew(r), r = [e, 0, 0]
    M[1, 5] = -config.mu * e   # F_v ↔ M_θz
    M[5, 1] = -config.mu * e
    M[2, 4] = +config.mu * e   # F_w ↔ M_θy
    M[4, 2] = +config.mu * e

    # Rotational inertias AT EA (CG values + Huygens-Steiner where needed)
    # config.radius_gyration is r_α² in [m²], NOT a dimensionless ratio
    # Polar inertia per unit span: I_p = μ × r_α²
    # Note: r_α² already includes geometric scaling, do NOT multiply by b²
    r2 = config.radius_gyration       # value in unit of semichord (nondimensional)
    b = config.chord / 2              # semichord
    r2 = (r2 *b)**2                     # DImensionally consistent with μ in kg/m → I_p in kg·m
    thickness = config.thickness_factor * config.chord

    i11 = config.mu * r2              # [kg·m]  polar inertia about elastic axis (torsion)
    i22 = config.k_i22 * i11          # [kg·m]  scaled component (chordwise bending)
    i33 = config.k_i33 * i11          # [kg·m]  scaled component (vertical bending)
    #i22 = config.mu * (thickness / 2)**2 / 12
    #i33 = config.mu * (config.chord)**2 / 12

    M[3, 3] = i22                        # θx (chordwise bending)
    M[4, 4] = i11 + config.mu * e**2                       # θy (torsion about span)
    M[5, 5] = i33 + config.mu * e**2                      # θz (vertical bending)

    # Reference layout: always use pitch=0 in create_beam_model; blade inclination
    # is applied afterward as a rigid rotation of the whole beam model about +X.
    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements, 0,
        agard_theory=False, flutter_benchmark=True, sonata=False,
        xea=xea, xcm=xcm, case_name='ABRAMSON1965'
    )

    alpha_deg = float(getattr(config, "alpha_deg", 0.0))
    beam_model = rotate_beam_model_y(beam_model, [alpha_deg])

    # Section 6×6 and mesh stay in the reference frame (beam along +Y after create);
    # structural pitch about +X is applied after global K/M and aerogrid are built
    # (see post_pitch_utils.apply_structural_pitch_about_x in main.py).
    beam_model["K_section"] = K
    beam_model["M_section"] = M
    if abs(alpha_deg) > 1e-12:
        print(f"  ABRAMSON1965: applied alpha {alpha_deg:.4f} deg about +Y on beam model (reference build).")
    
    # Optionally save the section stiffness and mass matrices to CSV files.
    # Controlled by config.save_matrices (bool). If True, matrices are written
    # to config.matrices_dir if provided, otherwise to ./matrices/.
    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            K_save = beam_model["K_section"]
            M_save = beam_model["M_section"]
            np.savetxt(k_file, K_save, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M_save, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model


def _build_goland_model(config):
    
    from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y
    from FEA.fea_utl.beam_properties import create_beam_model
    
    print(f"Building GOLAND structural model")

    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis
    # u: displacement in X, v: displacement in Y (axial), w: displacement in Z
    # θx: rotation about X, θy: rotation about Y (torsion), θz: rotation about Z
    K = np.zeros((6, 6), float)
    K[0, 0] = config.GAx # Shear stiffness X (transverse)
    K[1, 1] = config.EA  # Axial stiffness along Y (beam axis)
    K[2, 2] = config.GAz # Shear stiffness Z (transverse)
    K[3, 3] = config.EIxx # Bending stiffness about X
    K[4, 4] = config.GJ  # Torsional stiffness (rotation along Y - beam axis)
    K[5, 5] = config.EIzz # Bending stiffness about Z

    xcm = config.xcm_factor * config.chord
    xea = config.xea_factor * config.chord
    e = (xcm - xea)  # Offset in X direction: positive if CG is aft of EA

    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis (spanwise)
    M = np.zeros((6, 6), float)
    M[0, 0] = config.mu  # mass for u (chordwise X)
    M[1, 1] = config.mu  # mass for v (axial Y, span)
    M[2, 2] = config.mu  # mass for w (vertical Z)

    # Off-diagonal coupling from -mu*skew(r), r = [e, 0, 0]
    M[1, 5] = -config.mu * e   # F_v ↔ M_θz
    M[5, 1] = -config.mu * e
    M[2, 4] = +config.mu * e   # F_w ↔ M_θy  (KEY coupling for flutter: bending-torsion)
    M[4, 2] = +config.mu * e

    # Rotational inertias AT EA (CG values + Huygens-Steiner where needed)
    M[3, 3] = config.i22                         # θx (chordwise bending) - no PAT
    M[4, 4] = config.i11 + config.mu * e**2      # θy (torsion about span) + PAT
    M[5, 5] = config.i33 + config.mu * e**2      # θz (vertical bending)  + PAT

    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements, config.pitch,
        agard_theory=False, flutter_benchmark=True, sonata=False
    )
    
    beam_model = rotate_beam_model_y(beam_model, [config.alpha_deg])
    
    # Add matrices to the model for later use
    beam_model['K_section'] = K
    beam_model['M_section'] = M
    
    # Optionally save the section stiffness and mass matrices to CSV files.
    # Controlled by config.save_matrices (bool). If True, matrices are written
    # to config.matrices_dir if provided, otherwise to ./matrices/.
    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            np.savetxt(k_file, K, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model


def _build_gridconv_model(config):
    
    from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y
    from FEA.fea_utl.beam_properties import create_beam_model
    
    print(f"Building GOLAND structural model")

    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis
    K = np.zeros((6, 6), float)
    K[0, 0] = config.GAy      # Shear stiffness X (was GAy, transverse)
    K[1, 1] = config.EA       # Axial stiffness along Y (beam axis)
    K[2, 2] = config.GAz      # Shear stiffness Z (transverse)
    K[3, 3] = config.EIxx     # Bending stiffness about X
    K[4, 4] = config.GJ       # Torsional stiffness about Y (beam axis)
    K[5, 5] = config.EIzz*100 # Bending stiffness about Z (scaled)

    xcm = config.xcm_factor * config.chord
    xea = config.xea_factor * config.chord
    e = (xcm - xea)  # Fixed: was (xea - xcm) which gave wrong sign!

    # DOF ordering: [u, v, w, θx, θy, θz] where Y is the beam axis
    M = np.zeros((6, 6), float)
    M[0, 0] = config.mu  # mass for u displacement
    M[1, 1] = config.mu  # mass for v displacement (axial)
    M[2, 2] = config.mu  # mass for w displacement

    ## Coupling between w (transverse Z) and θy (torsion about Y-beam axis)
    #M[2, 4] = config.mu * e
    #M[4, 2] = config.mu * e
    #
    ## Rotational inertias
    #M[3, 3] = config.i22  # Inertia about X
    #M[4, 4] = config.i11  # Inertia about Y (torsion)
    #M[5, 5] = config.i33  # Inertia about Z

    # Off-diagonal coupling from -mu*skew(r), r = [e, 0, 0]
    M[1, 5] = -config.mu * e   # F_v ↔ M_θz
    M[5, 1] = -config.mu * e
    M[2, 4] = +config.mu * e   # F_w ↔ M_θy  (KEY coupling for flutter: bending-torsion)
    M[4, 2] = +config.mu * e

    # Rotational inertias AT EA (CG values + Huygens-Steiner where needed)
    # config.i22 = 0.1*i11 = inertia about X at CG → no PAT (offset // X)
    # config.i11 = 8.64    = inertia about Y at CG → PAT: +mu*e²
    # config.i33 = 0.9*i11 = inertia about Z at CG → PAT: +mu*e²
    M[3, 3] = config.i22                         # θx (chordwise bending) - no PAT
    M[4, 4] = config.i11 + config.mu * e**2      # θy (torsion about span) + PAT
    M[5, 5] = config.i33 + config.mu * e**2 
 

    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements, config.pitch,
        agard_theory=False, flutter_benchmark=True, sonata=False, center_beam=True, chord=config.chord
    )
    
    beam_model = rotate_beam_model_y(beam_model, [config.alpha_deg])
    
    # Add matrices to the model for later use
    beam_model['K_section'] = K
    beam_model['M_section'] = M
    
    # Optionally save the section stiffness and mass matrices to CSV files.
    # Controlled by config.save_matrices (bool). If True, matrices are written
    # to config.matrices_dir if provided, otherwise to ./matrices/.
    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            np.savetxt(k_file, K, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model


def _build_sonata_model(config):
    # Add the specific path for the correct parser to avoid ambiguity
    from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y
    from FEA.fea_utl.beam_model import create_beam_model
    
    parser_path = os.path.join(config.paths['SONATA'], config.sonata_name, 'csv_export')
    if parser_path not in sys.path:
        sys.path.insert(0, parser_path)
    
    from parser import parse_sectional_matrix_csv, transform_matrices

    # SONATA outputs
    K_dir = config.paths['SONATA'] + '/' + config.sonata_name + '/csv_export/' + config.sonata_case_name + '_anbax_beam_properties_stiff_matrices.csv'
    M_dir = config.paths['SONATA'] + '/' + config.sonata_name + '/csv_export/' + config.sonata_case_name + '_anbax_beam_properties_mass_matrices.csv'
    sections_props_csv = config.paths['SONATA'] + '/' + config.sonata_name + '/csv_export/' + config.sonata_case_name + '_section_data.csv'

    K = parse_sectional_matrix_csv(K_dir)
    M = parse_sectional_matrix_csv(M_dir)

    K, M = transform_matrices(sections_props_csv, K, M)

    # Check if the config specifies center_beam (default to False to keep original SONATA coords)
    center_beam = getattr(config, 'center_beam', False)
    chord = getattr(config, 'chord', None)
    
    beam_model = create_beam_model(
        K, M,
        config.n_elements, 
        sections_props_csv=sections_props_csv,
        center_beam=center_beam,
        chord=chord
    )
    
    beam_model = rotate_beam_model_y(beam_model, [config.alpha_deg])

    beam_model['K_section'] = K
    beam_model['M_section'] = M

    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            np.savetxt(k_file, K, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model


def _build_tnz_multibody_model(config):
    """
    Builds the combined arm + foil_wing multibody.

    The structure is:
      1. **tnz_arm_spline** sub-model:  a curved arm beam with mechanical 
         properties from `foil_tnz.py` sections, following the 3D spline 
         curve defined by cg_sc coordinates. Its tip connects to the foil junction.
      2. **foil_dx + foil_sx** sub-models: two horizontal foil beams with rigid properties.

    The arm root (top) is clamped.  The arm tip = foil junction at [0, 0, -0.3] m.
    Both foils (foil_dx and foil_sx) start at [0, 0, -0.3] and extend horizontally.

    The arm follows the 3D spline curve defined in foil_tnz.py sections via
    the cg_sc [x, y, z] coordinates (in mm), converted to meters.
    """
    from SONATA.ETNZ.tnz_arm.foil_tnz import sections as tnz_sections
    #from FEA.hydrofoils.foil_tnz import sections as tnz_sections
    from FEA.fea_utl.multibody_assembly import assemble_multibody
    from FEA.fea_utl.rbe2_connector import assemble_rbe2_to_model, validate_rbe2_connectivity
    
    try:
        from scipy.interpolate import CubicSpline
        use_scipy = True
    except ImportError:
        use_scipy = False
        print("  Warning: scipy not available, using linear interpolation for arm spline")

    print("Building tnz_multibody model (arm_spline + foil_dx + foil_sx, NO boot)")

    mm2m        = 1e-3
    Nmm2_to_Nm2 = 1e-6

    # ==================================================================
    # ------------------------------------------------------------------
    # 1. BUILD THE ARM SUB-MODEL
    # ------------------------------------------------------------------
    # ==================================================================

    valid = sorted(
        [s for s in tnz_sections],   # include all sections (spine_mm=-500 = "Top Bearing" through tip)
        key=lambda s: s["spine_mm"]
    )

    # Extract section properties and 3D coordinates
    spine_m   = np.array([s["spine_mm"] * mm2m                 for s in valid])
    EA_arr   = np.array([s["EA_N"]                            for s in valid])
    EIxx_arr = config.EIxx_scale * np.array([s["EIxx_Nmm2"] * Nmm2_to_Nm2        for s in valid])
    EIyy_arr = config.EIyy_scale * np.array([s["EIyy_Nmm2"] * Nmm2_to_Nm2        for s in valid])
    GK_arr   = config.GK_scale * np.array([s["GK_Nmm2"]   * Nmm2_to_Nm2        for s in valid])
    
    # TNZ case: shear center (elastic axis) coincides with center of gravity
    # e = offset between CG and SC (in X direction, chordwise)
    # Since SC = CG for TNZ, e should be zero everywhere
    e_arr = np.zeros_like(spine_m)  # e = 0 for all sections (SC ≡ CG)
    
    # Extract 3D spline coordinates (cg_sc = [x, y, z] in mm)
    cg_sc_x = np.array([s["cg_sc"][0] * mm2m for s in valid])  # chordwise
    cg_sc_y = np.array([s["cg_sc"][1] * mm2m for s in valid])  # lateral
    cg_sc_z = np.array([s["cg_sc"][2] * mm2m for s in valid])  # vertical

    arm_length = float(spine_m[-1] - spine_m[0])   # 4.2 m arc length
    # Discretization rule (IMPORTANT): the arm uses exactly config.n_elements.
    # Foils are controlled separately via n_nodes_foil.
    n_arm = int(getattr(config, 'n_elements'))

    # Create interpolators for 3D curve parametrized by spine coordinate
    # Node positions along the spline
    spine_nodes = np.linspace(spine_m[0], spine_m[-1], n_arm + 1)
    
    if use_scipy:
        # Use cubic spline interpolation for smooth curve
        interp_x = CubicSpline(spine_m, cg_sc_x, extrapolate=True)
        interp_y = CubicSpline(spine_m, cg_sc_y, extrapolate=True)
        interp_z = CubicSpline(spine_m, cg_sc_z, extrapolate=True)
        node_coords_x = interp_x(spine_nodes)
        node_coords_y = interp_y(spine_nodes)
        node_coords_z = interp_z(spine_nodes)
    else:
        # Use linear interpolation as fallback
        node_coords_x = np.interp(spine_nodes, spine_m, cg_sc_x)
        node_coords_y = np.interp(spine_nodes, spine_m, cg_sc_y)
        node_coords_z = np.interp(spine_nodes, spine_m, cg_sc_z)
    
    # Element midpoint positions for property interpolation
    spine_mid = 0.5 * (spine_nodes[:-1] + spine_nodes[1:])
    
    EA_el   = np.interp(spine_mid, spine_m, EA_arr)
    EIxx_el = np.interp(spine_mid, spine_m, EIxx_arr)
    EIyy_el = np.interp(spine_mid, spine_m, EIyy_arr)
    GK_el   = np.interp(spine_mid, spine_m, GK_arr)
    e_el = np.interp(spine_mid, spine_m, e_arr)  # Offset between CG and SC (= 0 for TNZ)

    # ------------------------------------------------------------------
    # Section areas and derived properties
    # ------------------------------------------------------------------
    nu_EG  = getattr(config, 'nu_EG')    # E/G ratio (CFRP laminate default)
    kappa  = getattr(config, 'kappa')   # Timoshenko shear correction factor
    rho_s  = getattr(config, 'rho_s')    # effective material density [kg/m³]

    section_area_mm2 = np.array([s["arm_max_thick_mm"] * s["arm_width_mm"] for s in valid])
    section_area_m2  = section_area_mm2 * mm2m**2          # [m²]
    
    # Compute mu distribution: linear mass density = density × cross-sectional area
    mu_arr = rho_s * section_area_m2  # [kg/m] mass per unit length at each section
    
    # Interpolate mu at element midpoints for distributed mass
    mu_el = np.interp(spine_mid, spine_m, mu_arr)  # [kg/m] mass density at element midpoints
    
    # Fallback to constant mu_arm if provided (for backward compatibility)
    if hasattr(config, 'mu_arm_constant') and config.mu_arm_constant:
        mu_el[:] = getattr(config, 'mu_arm', np.mean(mu_el))
    
    E_arr            = EA_arr / section_area_m2             # Young's modulus [Pa]
    G_arr            = E_arr  / nu_EG                      # shear modulus   [Pa]
    GAx_arr          = kappa * G_arr * section_area_m2     # shear stiffness [N]

    # Interpolate section-derived quantities at element midpoints
    E_el    = np.interp(spine_mid, spine_m, E_arr)
    area_el = np.interp(spine_mid, spine_m, section_area_m2)
    GAx_el  = np.interp(spine_mid, spine_m, GAx_arr)
    GAz_el  = GAx_el.copy()
    
    # Make GAx and GAz large so natural frequencies depend primarily on GJ (torsional stiffness)
    GAx_scale = getattr(config, 'GAx_scale', 1000)  # Default scaling factor
    GAz_scale = getattr(config, 'GAz_scale', 1000)  # Default scaling factor
    GAx_el *= GAx_scale
    GAz_el *= GAz_scale

    # Second moments of area from bending stiffness [m⁴]
    Ixx_area = EIxx_el / E_el
    Iyy_area = EIyy_el / E_el

    # Mass moments of inertia per unit length [kg·m]  (i = ρ·I_area)
    i11_el = rho_s * Iyy_area          # about local X (chordwise bending)
    i33_el = rho_s * Ixx_area          # about local Z (vertical bending)
    J_area  = Ixx_area + Iyy_area
    i22_el  = rho_s * J_area           # about local Y (torsion / polar)

    # Print summary for verification
    mid_idx = n_arm // 2
    print(f"  Arm section properties at midspan (spine={spine_mid[mid_idx]:.3f} m):")
    print(f"    E = {E_el[mid_idx]:.3e} Pa,  A = {area_el[mid_idx]:.4f} m²")
    print(f"    GA = {GAx_el[mid_idx]:.3e} N  (kappa={kappa}, G=E/{nu_EG:.1f})")
    print(f"    μ(spine) = ρ_s × A = {rho_s} kg/m³ × {area_el[mid_idx]:.4f} m² = {mu_el[mid_idx]:.1f} kg/m")
    print(f"    μ distribution: min={mu_el.min():.1f}, max={mu_el.max():.1f}, mean={mu_el.mean():.1f} kg/m")

    # Build per-element K and M matrices with distributed mu_el
    K_dict_arm = {}
    M_dict_arm = {}
    for i in range(n_arm):
        Ki = np.zeros((6, 6), float)
        Ki[0, 0] = GAx_el[i]   
        Ki[1, 1] = EA_el[i]
        Ki[2, 2] = GAz_el[i]   
        Ki[3, 3] = EIxx_el[i]
        Ki[4, 4] = GK_el[i]    
        Ki[5, 5] = EIyy_el[i]
        
        K_dict_arm[f"e{i}"] = Ki

        e = e_el[i]  # Offset between CG and SC (zero for TNZ case)
        mu_i = mu_el[i]  # Use distributed mass at element i
        Mi = np.zeros((6, 6), float)
        Mi[0, 0] = mu_i; 
        Mi[1, 1] = mu_i; 
        Mi[2, 2] = mu_i
        Mi[2, 4] = +mu_i * e;  
        Mi[4, 2] = +mu_i * e
        Mi[1, 5] = -mu_i * e;  
        Mi[5, 1] = -mu_i * e
        Mi[3, 3] = i11_el[i]
        Mi[4, 4] = i22_el[i] + mu_i * e**2
        Mi[5, 5] = i33_el[i] + mu_i * e**2
        
        M_dict_arm[f"e{i}"] = Mi

    # Build arm beam model with custom node positions following spline curve

    nodes_arm = []
    for i in range(n_arm + 1):
        # cg_sc coordinates are already converted from mm to m above
        nodes_arm.append({
            "position": [
                node_coords_x[i],  
                node_coords_y[i], 
                node_coords_z[i]
            ],
            "index": i
            ,"submodel": "arm_spline"
        })
    
    # Debug: print original tip position before translation
    tip_pos_original = np.array(nodes_arm[-1]["position"])
    root_pos_original = np.array(nodes_arm[0]["position"])
    print(f"  Arm original: root={root_pos_original}, tip={tip_pos_original}")

    # ------------------------------------------------------------------
    # Rotate the spine to align boot with vertical axis (Z) if specified in config.
    # Rotation matrix about X:
    _spine_rot_rad = np.deg2rad(config._spine_rot_deg)
    _c, _s = np.cos(_spine_rot_rad), np.sin(_spine_rot_rad)
    _Rx_spine = np.array([[1, 0,   0  ],
                          [0, _c, -_s ],
                          [0, _s,  _c ]])

    _pivot = np.array(nodes_arm[0]["position"], dtype=float)
    print(f"  Rotating spine by {config._spine_rot_deg}° around X-axis at pivot={_pivot}")
    
    for _node in nodes_arm:
        _p = np.array(_node["position"], dtype=float) - _pivot
        _p_rot = _Rx_spine @ _p
        _node["position"] = (_p_rot + _pivot).tolist()

    tip_pos_rotated  = np.array(nodes_arm[-1]["position"])
    root_pos_rotated = np.array(nodes_arm[0]["position"])
    print(f"  Arm after X-rotation: root={root_pos_rotated}, tip={tip_pos_rotated}")

    # Translate the entire arm so that the tip (last node) is at the foil root (boot end point) (foil_junction).
    _cg_file = os.path.join(config.paths['SONATA'], 'ETNZ', 'tnz_boot', 'CG_curve_points.py')
    _cg_spec = _ilu.spec_from_file_location('CG_curve_points', _cg_file)
    _cg_mod  = _ilu.module_from_spec(_cg_spec)
    _cg_spec.loader.exec_module(_cg_mod)
    _boot_tip_mm = np.array(_cg_mod.CG_points_tnz_boot[0], dtype=float)
    
    foil_junction = _boot_tip_mm * 1e-3   # mm → m
    tip_pos = np.array(nodes_arm[-1]["position"])
    for node in nodes_arm:
        node["position"] = [
            node["position"][0] - tip_pos[0] + foil_junction[0],
            node["position"][1] - tip_pos[1] + foil_junction[1],
            node["position"][2] - tip_pos[2] + foil_junction[2]
        ]
    
    # Debug: verify translation
    tip_pos_final = np.array(nodes_arm[-1]["position"])
    root_pos_final = np.array(nodes_arm[0]["position"])
    print(f"  Arm translated: root={root_pos_final}, tip={tip_pos_final}")
    
    # Build elements with actual lengths from curved geometry.
    elements_arm = []
    for i in range(n_arm):
        p1 = np.array(nodes_arm[i]["position"],   dtype=float)
        p2 = np.array(nodes_arm[i+1]["position"], dtype=float)
        diff = p2 - p1
        L_e  = float(np.linalg.norm(diff))

        # Tangent unit vector in global frame for this element
        tangent = diff / L_e

        # 6×6 rotation from local (+Y beam axis) to global (tangent direction)
        T6 = T6_from_beam_direction(tangent)

        K_loc = K_dict_arm[f"e{i}"]
        M_loc = M_dict_arm[f"e{i}"]

        # Rotate sectional matrices into the global frame
        K_glob = T6 @ K_loc @ T6.T
        M_glob = T6 @ M_loc @ T6.T

        elements_arm.append({
            "nodes":     [i, i + 1],
            "stiffness": K_glob,
            "mass":      M_glob,
            "length":    L_e,
            "T6":        T6,
            "beam_dir_global": tangent.tolist(),
        })
    
    sm_arm = {"nodes": nodes_arm, "elements": elements_arm}
    
    # Clamp at the node closest to spine_mm=0 ("Bot Bearing", second section in foil_tnz.py).
    # The arm now starts at spine_mm=-500 ("Top Bearing"), so the clamped node is
    # the first interpolated node whose spine coordinate is >= 0.
    clamp_spine_m = -500   # spine_mm=-500 in metres
    clamp_node_idx = int(np.argmin(np.abs(spine_nodes - clamp_spine_m)))
    sm_arm['nodes'][clamp_node_idx]['clamped'] = True
    print(f"  Arm: clamped node {clamp_node_idx} (spine≈{spine_nodes[clamp_node_idx]:.4f} m, "
          f"pos={sm_arm['nodes'][clamp_node_idx]['position']})")
    
    print(f"  Arm: {len(nodes_arm)} nodes along 3D spline curve")

    # ==================================================================
    # ------------------------------------------------------------------
    # 1. BUILD THE ARM SUB-MODEL
    # ------------------------------------------------------------------
    # ==================================================================

    # Compute total foil mass from config.mu (linear mass density)
    mu_foil_param = float(getattr(config, 'mu'))  # Will use this below after computing spans

    # Load foil endpoint positions from CG_curve_points files (mm → m).
    def _load_cg(subfolder, varname):
        _f   = os.path.join(config.paths['SONATA'], 'ETNZ', subfolder, 'CG_curve_points.py')
        _sp  = _ilu.spec_from_file_location('CG_curve_points_' + subfolder, _f)
        _mod = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_mod)
        return np.array(getattr(_mod, varname), dtype=float) * 1e-3   # mm → m

    cg_dx = _load_cg('tnz_foil_dx', 'CG_points_tnz_foil_dx')  # shape (2, 3)
    cg_sx = _load_cg('tnz_foil_sx', 'CG_points_tnz_foil_sx')  # shape (2, 3)

    foil_dx_root0 = cg_dx[0];  foil_dx_tip0 = cg_dx[1]   # [0]=junction, [1]=free tip
    foil_sx_root0 = cg_sx[1];  foil_sx_tip0 = cg_sx[0]   # [1]=junction, [0]=free tip

    foil_dx_vec0  = foil_dx_tip0 - foil_dx_root0
    foil_sx_vec0  = foil_sx_tip0 - foil_sx_root0

    L_foil_dx_nat = float(np.linalg.norm(foil_dx_vec0))
    L_foil_sx_nat = float(np.linalg.norm(foil_sx_vec0))

    L_foil_dx = float(getattr(config, 'foil_span', L_foil_dx_nat))
    L_foil_sx = float(getattr(config, 'foil_span', L_foil_sx_nat))

    foil_dx_dir0 = foil_dx_vec0 / L_foil_dx_nat
    foil_sx_dir0 = foil_sx_vec0 / L_foil_sx_nat

    # Compute total foil mass from config.mu (linear mass density)
    foil_mass_total = mu_foil_param * (L_foil_dx + L_foil_sx)
    
    # ------------------------------------------------------------------
    # Foil mass model selection
    # ------------------------------------------------------------------
    foil_mass_model = config.foil_mass_model

    # Defaults (safe for point_mass): no distributed mass, no taper, zero inertias
    use_distributed = False
    add_point_mass = False
    taper_p = 0.0
    chord_foil = float(getattr(config, 'chord', 0.4))   # [m]
    mu_foil_mean = 0.0
    i11_foil = 0.0
    i22_foil = 0.0
    i33_foil = 0.0

    if foil_mass_model == 'hybrid':
        use_distributed = True
        add_point_mass = True
        mu_foil_mean = mu_foil_param  # uniform distribution
        i_polar    = mu_foil_mean * chord_foil**2 / 12.0
        i11_foil   = i_polar / 2.0
        i22_foil   = i_polar
        i33_foil   = i_polar / 2.0
        taper_p = 0.0  # straight rectangular, no taper

        print("  Foil mass model: HYBRID (uniform distributed mu + point mass)")
        print(f"  Mean mu={mu_foil_mean:.4f} kg/m  "
              f"(total={mu_foil_mean*(L_foil_dx+L_foil_sx):.1f} kg, chord={chord_foil:.3f} m)")

    elif foil_mass_model == 'distributed':
        use_distributed = True
        # ── Total mass budget ──────────────────────────────────────────
        # mu_foil is the *mean* linear mass density [kg/m] that conserves total mass.
        mu_foil_mean = mu_foil_param   # Use config.mu directly

        # ── Rotational inertia per unit length (thin-strip polar approx) ──
        # Computed from the mean mu so that total rotational inertia is conserved.
        chord_foil = float(getattr(config, 'chord', 0.4))   # [m]
        i_polar    = mu_foil_mean * chord_foil**2 / 12.0
        i11_foil   = i_polar / 2.0   # about beam-local X  (flapwise)
        i22_foil   = i_polar          # about beam-local Y  (torsion / polar)
        i33_foil   = i_polar / 2.0   # about beam-local Z  (edgewise)

        # ── Load target CG from wings_point_mass.py ───────────────────
        _pm_file_tgt = os.path.join(config.paths['SONATA'], 'ETNZ',
                                    'tnz_point_mass', 'wings_point_mass.py')
        _pm_spec_tgt = _ilu.spec_from_file_location('wings_point_mass_tgt', _pm_file_tgt)
        _pm_mod_tgt  = _ilu.module_from_spec(_pm_spec_tgt)
        _pm_spec_tgt.loader.exec_module(_pm_mod_tgt)
        z_target = float(np.array(_pm_mod_tgt.wings_point_CG[0])[2]) * 1e-3  # mm → m

        z_root = float(foil_junction[2])   # Z of root node = −0.3 m

        # Compute post-dihedral Z-component of the foil beam direction.
        # The dihedral rotation for foil_dx is Rx(-θ) where θ = dihedral_angle.
        # _Rx(-θ) has Z-row = [0, -sin(θ), cos(θ)], so:
        #   Z_rotated = -sin(θ)*dir0[1] + cos(θ)*dir0[2]
        _dih_rad = np.deg2rad(float(getattr(config, 'dihedral_angle', 0.0)))
        _dz_rotated = (-np.sin(_dih_rad) * foil_dx_dir0[1]
                       + np.cos(_dih_rad) * foil_dx_dir0[2])
        dz_per_unit_arc = float(_dz_rotated)   # Z-component of post-dihedral beam direction

        if abs(dz_per_unit_arc) < 1e-9:
            # Horizontal foils — no Z variation along arc → uniform distribution
            taper_p = 0.0
            print(f"  Foil mass taper: foils are horizontal, using uniform distribution (p=0)")
        else:
            xi_target = (z_target - z_root) / (dz_per_unit_arc * L_foil_dx)
            xi_target = float(np.clip(xi_target, 1e-6, 1.0 - 1e-6))
            taper_p = 1.0 / xi_target - 2.0   # exact closed-form solution

            if taper_p < 0.0:
                print(f"  ⚠  Target CoM at ξ={xi_target:.4f} > 0.5 (tip-heavy); "
                      f"cannot achieve with root-heavy profile. Using uniform (p=0).")
                taper_p = 0.0

            z_com_check = z_root + (1.0 / (taper_p + 2.0)) * dz_per_unit_arc * L_foil_dx
            print(f"  Foil mass taper (power-law): p={taper_p:.4f}  "
                  f"(ξ*={xi_target:.4f}, z*_com={z_com_check:.4f} m, target={z_target:.4f} m)")

        print(f"  Foil mass model: DISTRIBUTED (power-law tapered, p={taper_p:.4f})")
        print(f"  Mean mu={mu_foil_mean:.4f} kg/m  "
              f"(total={mu_foil_mean*(L_foil_dx+L_foil_sx):.1f} kg, chord={chord_foil:.3f} m)")
        print(f"  Foil rotational inertia: i11={i11_foil:.5f}, i22={i22_foil:.5f}, "
              f"i33={i33_foil:.5f} kg·m")

    elif foil_mass_model == 'point_mass':
        # Foil beams are structurally massless; all 849 kg lives at the exact
        # CG position from wings_point_mass.py (a ghost node, not on the FEM mesh).
        add_point_mass = True
        mu_foil_mean = 0.0
        taper_p      = 0.0
        print(f"  Foil mass model: POINT_MASS  "
              f"(foil beams massless; {foil_mass_total:.1f} kg at exact CG)")

    else:
        raise ValueError(f"Unknown foil_mass_model='{foil_mass_model}'. "
                         "Choose 'distributed', 'point_mass', or 'hybrid'.")

    # ------------------------------------------------------------------
    # Foil structural representation (TNZ multibody)
    # ------------------------------------------------------------------
    # IMPORTANT:
    #  - We DO NOT create flexible (or artificially stiff) foil beam elements here.
    #  - We only create the foil *nodes* (discretized along the foil span).
    #  - Those foil nodes will be attached rigidly to the arm tip via RBE2 links.
    #
    # This prevents the foils from contributing their own bending/torsional stiffness
    # (which was previously happening via extremely stiff beam elements) and ensures
    # that only the rigid-body kinematics (through RBE2 constraints) is present.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Apply dihedral rotation around foil_junction.
    dihedral_deg = getattr(config, 'dihedral_angle')
    dihedral_rad = np.deg2rad(dihedral_deg)

    def _Rx(theta):
        """3×3 rotation matrix about the X-axis by angle theta [rad]."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]])

    Rx_dx = _Rx(-dihedral_rad)
    Rx_sx = _Rx(dihedral_rad)  # Opposite sign for the other foil to maintain symmetry

    foil_dx_dir = Rx_dx @ foil_dx_dir0
    foil_sx_dir = Rx_sx @ foil_sx_dir0

    print(f"  Dihedral angle: {dihedral_deg:.2f}°")
    print(f"  foil_dx natural dir: {foil_dx_dir0}  →  after dihedral: {foil_dx_dir}")
    print(f"  foil_sx natural dir: {foil_sx_dir0}  →  after dihedral: {foil_sx_dir}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Foil nodes discretization (nodes-only rigid representation)
    # ------------------------------------------------------------------
    # Each foil will have n_nodes_foil rigid nodes (so n_nodes_foil-1 segments).
    # This is intentionally decoupled from config.n_elements (arm discretization).
    n_nodes_foil = int(getattr(config, 'n_nodes_foil', 6))
    if n_nodes_foil < 2:
        raise ValueError(f"n_nodes_foil must be >= 2, got {n_nodes_foil}")

    # Keep a per-foil element count variable only for internal spacing.
    n_foil_dx = n_nodes_foil - 1
    n_foil_sx = n_nodes_foil - 1
    print(f"  Foil rigid-node discretization: n_nodes_foil={n_nodes_foil} per side "
          f"(segments dx={n_foil_dx}, sx={n_foil_sx})")

    def _make_foil_nodes_only_submodel(n_el, L, beam_dir_global, origin_global, name):
        """Create a nodes-only foil submodel (no elements)."""
        n_nodes = n_el + 1
        xi = np.linspace(0.0, 1.0, n_nodes)
        nodes = []
        origin_global = np.array(origin_global, dtype=float)
        beam_dir_global = np.array(beam_dir_global, dtype=float)
        for i in range(n_nodes):
            p = origin_global + (xi[i] * L) * beam_dir_global
            nodes.append({
                'position': p.tolist(),
                'index': i,
                'submodel': name,
            })
        return {'nodes': nodes, 'elements': []}

    sm_foil_dx = _make_foil_nodes_only_submodel(n_foil_dx, L_foil_dx, foil_dx_dir, foil_junction, 'foil_dx')
    sm_foil_sx = _make_foil_nodes_only_submodel(n_foil_sx, L_foil_sx, foil_sx_dir, foil_junction, 'foil_sx')

    # ------------------------------------------------------------------
    # Optional chordwise offset of the foil assembly
    # ------------------------------------------------------------------
    # `foil_chordwise_offset` [m] moves both foil sub-models forward (+X)
    # or backward (-X) relative to the arm tip (foil_junction).
    # When non-zero a rigid connector beam is inserted from foil_junction
    # to the new foil root so that the assembly remains connected.
    foil_chordwise_offset = float(getattr(config, 'foil_chordwise_offset'))

    if abs(foil_chordwise_offset) > 1e-9:
        offset_vec = np.array([foil_chordwise_offset, 0.0, 0.0], dtype=float)
        new_foil_junction = foil_junction + offset_vec

        print(f"\n  Foil chordwise offset: {foil_chordwise_offset:+.4f} m  "
              f"({foil_chordwise_offset*1e3:+.1f} mm in X)")
        print(f"  Original foil_junction:  {foil_junction}")
        print(f"  Shifted foil root:        {new_foil_junction}")

        # Shift all nodes of both foil node-sets by offset_vec
        for sm in (sm_foil_dx, sm_foil_sx):
            for node in sm['nodes']:
                p = np.array(node['position'], dtype=float)
                node['position'] = (p + offset_vec).tolist()

        # Build a rigid connector element: foil_junction → new_foil_junction.
        # This stub beam is a single-element sub-model with 2 nodes.
        conn_vec = offset_vec
        conn_len = float(np.linalg.norm(conn_vec))
        conn_dir = conn_vec / conn_len                      # unit direction (+X or -X)
        T6_conn  = T6_from_beam_direction(conn_dir)

        K_conn_loc = np.zeros((6, 6), float)
        for _d in range(6):
            K_conn_loc[_d, _d] = 1e12   # rigid in all DOFs
        M_conn_loc = np.zeros((6, 6), float)   # massless connector

        K_conn_glob = T6_conn @ K_conn_loc @ T6_conn.T
        M_conn_glob = np.zeros((6, 6), float)

        conn_nodes = [
            {'position': foil_junction.tolist(),    'index': 0},
            {'position': new_foil_junction.tolist(), 'index': 1},
        ]
        conn_elem = [{
            'nodes':           [0, 1],
            'stiffness':       K_conn_glob,
            'mass':            M_conn_glob,
            'length':          conn_len,
            'T6':              T6_conn,
            'beam_dir_global': conn_dir.tolist(),
            'rigid_link':      True,
        }]
        sm_connector = {'nodes': conn_nodes, 'elements': conn_elem}
        print(f"  Rigid connector: foil_junction → new_foil_junction  "
              f"(L={conn_len:.4f} m, dir={conn_dir})")

        # The foil_junction node on the connector will be merged with the arm
        # tip; the opposite end merges with both foil roots during assembly.
        sm_foil_parts = [sm_connector, sm_foil_dx, sm_foil_sx]

        # Update foil_junction reference for CG / debug reporting below
        foil_junction_eff = new_foil_junction
    else:
        sm_foil_parts = [sm_foil_dx, sm_foil_sx]
        foil_junction_eff = foil_junction

    print(f"\n  === DEBUG: Foil node positions ===")
    print(f"  foil_dx: root={sm_foil_dx['nodes'][0]['position']}, tip={sm_foil_dx['nodes'][-1]['position']}")
    print(f"  foil_sx: root={sm_foil_sx['nodes'][0]['position']}, tip={sm_foil_sx['nodes'][-1]['position']}")

    # ------------------------------------------------------------------
    # Centre of mass of the foil assembly (for reporting/debug)
    # ------------------------------------------------------------------
    # Now that foils are modeled as *nodes only*, we approximate the CoM using
    # the chosen mass model:
    #   - distributed/hybrid: assume linear density profile along span and
    #     place element masses at segment midpoints
    #   - point_mass: CoM is not on the mesh; we report geometric centroid here
    def _foil_com_nodes_only(sm, L, mu_mean, p):
        """Return (total_mass [kg], com [3]) for a nodes-only foil representation."""
        nodes = sm['nodes']
        n_el = max(len(nodes) - 1, 0)
        if n_el == 0:
            return 0.0, np.zeros(3, float)

        if mu_mean <= 1e-12:
            # massless (point_mass case)
            return 0.0, np.zeros(3, float)

        # mu(s) = mu0 * (1 - s/L)^p, with mu0 = mu_mean*(p+1)
        mu0 = mu_mean * (p + 1.0)
        total_m = 0.0
        weighted_pos = np.zeros(3, float)
        for i in range(n_el):
            p1 = np.array(nodes[i]['position'], float)
            p2 = np.array(nodes[i + 1]['position'], float)
            # segment arc-length (should be ~L/n_el even after dihedral)
            seg_len = float(np.linalg.norm(p2 - p1))
            s_mid = (i + 0.5) * (L / n_el)
            mu_mid = mu0 * max(1.0 - s_mid / L, 0.0) ** p
            m_seg = mu_mid * seg_len
            total_m += m_seg
            weighted_pos += m_seg * 0.5 * (p1 + p2)

        if total_m > 1e-12:
            weighted_pos /= total_m
        return total_m, weighted_pos

    m_dx, com_dx = _foil_com_nodes_only(sm_foil_dx, L_foil_dx, mu_foil_mean, taper_p)
    m_sx, com_sx = _foil_com_nodes_only(sm_foil_sx, L_foil_sx, mu_foil_mean, taper_p)
    m_foils_total = m_dx + m_sx

    if m_foils_total > 1e-12:
        com_foils = (m_dx * com_dx + m_sx * com_sx) / m_foils_total
    else:
        # Massless foils (point_mass model) — compute geometric centroid instead
        com_foils = 0.25 * (
            np.array(sm_foil_dx['nodes'][0]['position'], float) +
            np.array(sm_foil_dx['nodes'][-1]['position'], float) +
            np.array(sm_foil_sx['nodes'][0]['position'], float) +
            np.array(sm_foil_sx['nodes'][-1]['position'], float)
        )

    print(f"\n  === Foil assembly centre of mass ===")
    print(f"  foil_dx:  mass={m_dx:.2f} kg,  CoM={com_dx} m")
    print(f"  foil_sx:  mass={m_sx:.2f} kg,  CoM={com_sx} m")
    print(f"  Combined: mass={m_foils_total:.2f} kg,  CoM={com_foils} m")

    # Load the reference point-mass CG and compare
    _pm_file_cg = os.path.join(config.paths['SONATA'], 'ETNZ',
                               'tnz_point_mass', 'wings_point_mass.py')
    _pm_spec_cg = _ilu.spec_from_file_location('wings_point_mass_cg', _pm_file_cg)
    _pm_mod_cg  = _ilu.module_from_spec(_pm_spec_cg)
    _pm_spec_cg.loader.exec_module(_pm_mod_cg)
    cg_ref_mm = np.array(_pm_mod_cg.wings_point_CG[0], dtype=float)
    cg_ref_m  = cg_ref_mm * 1e-3

    delta_cg = com_foils - cg_ref_m
    print(f"  Reference CG (wings_point_mass.py): {cg_ref_m} m")
    print(f"  Δ CoM vs reference: {delta_cg} m  "
          f"(|Δ|={np.linalg.norm(delta_cg)*1e3:.1f} mm)")

    # Store on the model for post-processing
    beam_model_meta = {
        'foil_com':        com_foils,
        'foil_mass_total': m_foils_total,
        'foil_cg_ref_m':   cg_ref_m,
    }

    # ------------------------------------------------------------------
    # 3. Assemble arm + (optional rigid connector) + foil_dx + foil_sx
    # Two assembly strategies available:
    #   (a) Node merging (default): arm tip and foil roots are coincident → nodes merged
    #   (b) RBE2 connectors: arm tip (master) rigidly connected to foil roots (slaves)
    #       via high-stiffness rigid link elements, preserving distinct nodes
    # ------------------------------------------------------------------
    # In tnz_multibody we *must* use RBE2 when foils are nodes-only,
    # otherwise the nodes would be floating (no elements to connect them).
    use_rbe2 = True
    
    # SAVE LAST ARM NODE POSITION BEFORE ASSEMBLY (for point mass anchor)
    last_arm_node_pos_before_assembly = np.array(sm_arm['nodes'][-1]['position'], dtype=float)
    
    if use_rbe2:
        print(f"\n  === RBE2 Connector Mode ===")
        print(f"  Using RBE2 rigid connectors between arm tip and foil roots")
        
        # Assemble WITHOUT merging: keep arm and foils as separate sub-models
        # with distinct nodes at the junction
        beam_model = assemble_multibody([sm_arm] + sm_foil_parts, tol=1e-7)
        
        # Find arm tip node index (last node of the arm, before foil nodes are added)
        arm_tip_idx = len(sm_arm['nodes']) - 1
        
        # ------------------------------------------------------------------
        # RBE2 Configuration: Connect ALL foil nodes to arm tip
        # Master: arm_tip_idx (reference node)
        # Slaves: ALL foil nodes (both foil_dx and foil_sx) 
        # Result: Entire foil assembly moves rigidly with arm tip
        # ------------------------------------------------------------------
        
        # After assembly, we need to identify which nodes belong to the foils
        # The assembled model has:
        # - Nodes 0 to (len(sm_arm['nodes'])-1): arm nodes
        # - Nodes len(sm_arm['nodes']) onwards: foil nodes
        # 
        # But nodes at the root of each foil have been merged with the arm tip,
        # so we need to collect remaining foil nodes carefully.
        
        n_arm_nodes = len(sm_arm['nodes'])
        n_assembled_nodes = len(beam_model['nodes'])
        
        # All nodes after the arm are foil nodes.
        # Using assemble_multibody with tol=1e-7 and nodes-only submodels,
        # we expect *no* merging at the junction, so foil roots remain distinct.
        all_foil_node_indices = list(range(n_arm_nodes, n_assembled_nodes))
        
        print(f"\n  === RBE2 Full Assembly Connectivity ===")
        print(f"  Arm nodes: 0 to {n_arm_nodes - 1} (total: {n_arm_nodes})")
        print(f"  Master node (arm tip): {arm_tip_idx}")
        print(f"  Assembled model total nodes: {n_assembled_nodes}")
        print(f"  Foil slave nodes: {n_arm_nodes} to {n_assembled_nodes - 1} (total: {len(all_foil_node_indices)})")
        print(f"  Slave node indices (first 15): {all_foil_node_indices[:15]}{'...' if len(all_foil_node_indices) > 15 else ''}")
        
        # Create RBE2 connector from arm tip to ALL foil nodes
        rbe2_stiffness = getattr(config, 'rbe2_stiffness_scale', 1e12)
        assemble_rbe2_to_model(
            beam_model,
            master_node_idx=arm_tip_idx,
            slave_node_indices=all_foil_node_indices,
            rbe2_type='rigid_links',
            stiffness_scale=rbe2_stiffness,
            print_debug=False  # Will be very verbose with many nodes, so disable detailed output
        )
        
        # Validate RBE2 connectivity
        print(f"\n  === RBE2 Validation ===")
        validate_rbe2_connectivity(beam_model, verbose=True)
        
    else:
        # This branch is intentionally unreachable now.
        # It is kept only to minimize diffs with previous versions.
        raise RuntimeError(
            "tnz_multibody: foils are nodes-only; node-merging mode would leave them disconnected."
        )

    beam_model['K_section'] = list(K_dict_arm.values())[n_arm // 2]
    beam_model['M_section'] = list(M_dict_arm.values())[n_arm // 2]
    beam_model['is_multibody'] = True
    beam_model['sub_beam_names'] = (
        ['arm_spline', 'foil_connector', 'foil_dx', 'foil_sx']
        if abs(foil_chordwise_offset) > 1e-9
        else ['arm_spline', 'foil_dx', 'foil_sx']
    )
    # Helpful metadata for plotting/debug
    beam_model['arm_n_elements'] = int(n_arm)
    beam_model['arm_n_nodes'] = int(n_arm + 1)
    beam_model.update(beam_model_meta)   # foil_com, foil_mass_total, foil_cg_ref_m

    # ------------------------------------------------------------------
    # 4. Point-mass ghost node (for 'point_mass' and 'hybrid' foil_mass_model)
    # ------------------------------------------------------------------
    if add_point_mass:
        # Prefer explicit config overrides if provided; otherwise fall back to wings_point_mass.py
        pm_mass_cfg = getattr(config, 'foil_mass', None)
        pm_cg_cfg   = getattr(config, 'foil_mass_location', None)

        if pm_mass_cfg is not None:
            pm_mass = float(pm_mass_cfg)
        else:
            _pm_file = os.path.join(config.paths['SONATA'], 'ETNZ',
                                    'tnz_point_mass', 'wings_point_mass.py')
            _pm_spec = _ilu.spec_from_file_location('wings_point_mass', _pm_file)
            _pm_mod  = _ilu.module_from_spec(_pm_spec)
            _pm_spec.loader.exec_module(_pm_mod)
            pm_mass = float(_pm_mod.value)

        if pm_cg_cfg is not None:
            cg_arr = np.array(pm_cg_cfg, dtype=float)
            # Auto-convert mm → m if large values are passed (defensive for legacy data)
            if np.max(np.abs(cg_arr)) > 20.0:
                cg_arr = cg_arr * 1e-3
            cg_m = cg_arr
        else:
            _pm_file = os.path.join(config.paths['SONATA'], 'ETNZ',
                                    'tnz_point_mass', 'wings_point_mass.py')
            _pm_spec = _ilu.spec_from_file_location('wings_point_mass', _pm_file)
            _pm_mod  = _ilu.module_from_spec(_pm_spec)
            _pm_spec.loader.exec_module(_pm_mod)
            cg_mm  = np.array(_pm_mod.wings_point_CG[0], dtype=float)   # [mm]
            cg_m   = cg_mm * 1e-3                                       # → [m]

        # If the foils were shifted chordwise, translate the point mass by
        # the same offset so the CG tracks the foil assembly.
        # This behavior can be controlled by config.cg_wing_offset (default True).
        cg_wing_offset_flag = config.cg_wing_mass_offset_flag
        if abs(foil_chordwise_offset) > 1e-9 and cg_wing_offset_flag:
            cg_m = cg_m + np.array([foil_chordwise_offset, 0.0, 0.0], dtype=float)
            print(f"    Point-mass CG shifted by foil_chordwise_offset "
                  f"{foil_chordwise_offset:+.4f} m → new CG: {cg_m}")
        elif abs(foil_chordwise_offset) > 1e-9 and not cg_wing_offset_flag:
            print(f"    Point-mass CG NOT shifted (cg_wing_offset=False). "
                  f"Foil offset: {foil_chordwise_offset:+.4f} m, CG remains at: {cg_m}")

        # ── Anchor rigid link at the LAST ARM NODE ──
        # The last arm node was saved BEFORE assembly (last_arm_node_pos_before_assembly)
        # Find this node in the assembled mesh by position matching
        node_positions = np.array([n['position'] for n in beam_model['nodes']], dtype=float)
        
        # Find the node closest to the saved last arm node position
        dists_to_last_arm = np.linalg.norm(node_positions - last_arm_node_pos_before_assembly, axis=1)
        anchor_idx = int(np.argmin(dists_to_last_arm))
        anchor_pos = node_positions[anchor_idx]
        
        print(f"\n  Point mass anchor:")
        print(f"    Last arm node position (before assembly): {last_arm_node_pos_before_assembly}")
        print(f"    Anchor node index (after assembly): {anchor_idx}")
        print(f"    Anchor node position: {anchor_pos}")
        print(f"    Distance from saved position: {dists_to_last_arm[anchor_idx]:.6e} m")

        # Link goes from the last arm node to the CG of the point mass
        link_vec = cg_m - anchor_pos
        link_len = float(np.linalg.norm(link_vec))

        print(f"\n  Point-mass ghost node:")
        print(f"    mass = {pm_mass:.2f} kg")
        print(f"    CG position (m): {cg_m}")
        print(f"    Anchor (foil joint) node: {anchor_idx}  pos={anchor_pos}  dist={link_len:.4f} m")

        # ── Append ghost node ──────────────────────────────────────────
        ghost_idx = len(beam_model['nodes'])
        beam_model['nodes'].append({
            'position': cg_m.tolist(),
            'index':    ghost_idx,
        })

        # ── Build rigid link element (very high K, zero M) ─────────────
        # The link direction is from anchor to ghost node.
        if link_len < 1e-9:
            # CG coincides exactly with a FEM node — no link needed,
            # just place the point mass directly on the anchor node.
            print(f"    CG coincides with node {anchor_idx}; no rigid link inserted.")
            ghost_idx = anchor_idx   # mass will go on the existing node
            # Remove the ghost node we just appended
            beam_model['nodes'].pop()
        else:
            link_dir = (cg_m - anchor_pos) / link_len
            T6_link  = T6_from_beam_direction(link_dir)

            # Local sectional stiffness: axially very stiff, shear very stiff
            # Use 1e12 N/N·m² as "rigid" — orders of magnitude above foil stiffness
            K_link_loc = np.zeros((6, 6), float)
            for _d in range(6):
                K_link_loc[_d, _d] = 1e12    # rigid in all 6 local DOFs

            M_link_loc = np.zeros((6, 6), float)   # massless link

            K_link_glob = T6_link @ K_link_loc @ T6_link.T
            M_link_glob = np.zeros((6, 6), float)

            beam_model['elements'].append({
                'nodes':           [anchor_idx, ghost_idx],
                'stiffness':       K_link_glob,
                'mass':            M_link_glob,
                'length':          link_len,
                'T6':              T6_link,
                'beam_dir_global': link_dir.tolist(),
                'rigid_link':      True,     # marker for diagnostics
            })
            print(f"    Rigid link element appended: node {anchor_idx} → node {ghost_idx}  "
                  f"(L={link_len:.4f} m, dir={link_dir})")

        # ── Register point mass ────────────────────────────────────────
        # For a point mass connected via a rigid link of length r, assign
        # a small rotational inertia (solid-sphere approximation: I = 2/5 m r²)
        # to prevent zero-inertia rotational DOFs at the ghost node from
        # producing spurious modes in the eigenvalue problem.
        _r_link = link_len if link_len >= 1e-9 else float(getattr(config, 'chord', 0.4)) * 0.1
        _I_pt   = (2.0 / 5.0) * pm_mass * _r_link**2
        _I_diag = np.diag([_I_pt, _I_pt, _I_pt])

        beam_model['point_masses'] = [{
            'node_index': ghost_idx,
            'mass':       pm_mass,
            'inertia':    _I_diag,   # 3×3 diagonal rotational inertia
        }]
        print(f"    Point mass: {pm_mass:.1f} kg registered at node {ghost_idx} "
              f"(pos={beam_model['nodes'][ghost_idx]['position']})")
        print(f"    Rotational inertia (sphere approx, r={_r_link:.3f} m): "
              f"I={_I_pt:.4f} kg·m²")
    else:
        # No point masses requested
        beam_model['point_masses'] = []

    print(f"tnz_multibody assembled: {len(beam_model['nodes'])} nodes, "
          f"{len(beam_model['elements'])} elements")

    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               os.path.join(config.output_dir, config.name))
        os.makedirs(matrices_dir, exist_ok=True)

        model_name = config.name
        K = beam_model['K_section']
        M = beam_model['M_section']
        k_file = os.path.join(matrices_dir, f"{model_name}_K_section.csv")
        m_file = os.path.join(matrices_dir, f"{model_name}_M_section.csv")

        try:
            np.savetxt(k_file, K, delimiter=',', fmt='%.6e')
            np.savetxt(m_file, M, delimiter=',', fmt='%.6e')
            print(f"Saved section stiffness matrix to {k_file}")
            print(f"Saved section mass matrix to {m_file}")
        except Exception as _e:
            print(f"Warning: could not save section matrices to {matrices_dir}: {_e}")

    return beam_model