# FSI/Hydroelastic_analysis_workflow/structural_model.py

import numpy as np
import sys
import os
import importlib.util as _ilu
from FEA.fea_utl.multibody_assembly import (
    T6_from_beam_direction,
    orient_element_section_from_local,
    rotate_beam_model_to_global,
    translate_beam_model,
)
from FEA.fea_utl.beam_geometry import (
    aerogrid_xea_factor,
    beam_geometry_from_config,
    beam_on_shear_center,
    structural_xea_chordwise_offset,
    xea_decoupled_from_aerogrid,
)
from FEA.fea_utl.beam_properties import create_beam_model
from FEA.fea_utl.rotate_beams_y import rotate_beam_model_y

# Cases that use the uniform 6×6 custom section builder (unless overridden by config).
_CUSTOM_STRUCTURAL_MODEL_NAMES = frozenset({
    'ABRAMSON1965',
    'Model1',
    'Model2t',
    'NACA0015_LiHua2020',
})


def _uses_custom_structural_model(config) -> bool:
    """True when the case should use ``_build_structural_model_custom``."""
    builder = getattr(config, 'structural_model_type', None)
    if builder == 'custom':
        return True
    if builder is not None and builder != 'custom':
        return False
    return config.name in _CUSTOM_STRUCTURAL_MODEL_NAMES


def _apply_mass_cg_coupling(M, mu, e, config=None):
    """
    Off-diagonal translational–rotational coupling from CG offset ``e`` along chord.

    Default: M[2,4]=+μe, M[1,5]=-μe.
    ``flip_mass_cg_coupling_sign=True`` inverts both; ``flip_mass_cg_coupling_24_sign=True`` only M[2,4].
    """
    flip_both = config is not None and getattr(config, 'flip_mass_cg_coupling_sign', False)
    flip_24 = flip_both or (config is not None and getattr(config, 'flip_mass_cg_coupling_24_sign', False))
    flip_15 = flip_both
    c_wy = (-1.0 if flip_24 else 1.0) * (+float(mu) * float(e))
    c_vz = (-1.0 if flip_15 else 1.0) * (-float(mu) * float(e))
    M[1, 5] = c_vz
    M[5, 1] = c_vz
    M[2, 4] = c_wy
    M[4, 2] = c_wy


def _set_section_mass_at_reference(M, mu, e, *, i3, i4, i5, config=None):
    """6×6 mass matrix at the beam reference node (EA / shear center).

    Translational mass μ, rotational inertias ``i3,i4,i5`` on θx, θy, θz, and
    CG offset ``e`` [m] along chord (+x = aft of SC).

    Default (SHARPy Goland / EA-referenced): inertias are already at the
    reference; offset mass enters via skew coupling only (no +μe² PAT).

    Diagnostic: ``config.section_mass_add_mu_e2=True`` also adds μe² on θy/θz
    (legacy GINO PAT on top of EA-referenced i11/i33 — double-counts vs SHARPy).
    """
    M[0, 0] = mu
    M[1, 1] = mu
    M[2, 2] = mu
    _apply_mass_cg_coupling(M, mu, e, config)
    pat = bool(config is not None and getattr(config, "section_mass_add_mu_e2", False))
    pat_term = float(mu) * float(e) ** 2 if pat else 0.0
    M[3, 3] = i3
    M[4, 4] = i4 + pat_term
    M[5, 5] = i5 + pat_term
    if pat:
        print(
            f"  [section_mass_add_mu_e2] +μe²={pat_term:.6g} on θy/θz "
            f"(legacy PAT; not SHARPy EA convention)"
        )


def build(config):
    """
    Builds the structural beam model based on the provided configuration.
    """
    sys.path.extend([
        config.paths['FSI'],
        config.paths['SONATA'],
        config.paths['FEA']
    ])

    # Aggancio generico: un caso nuovo che arriva da csv_export si dichiara con
    # structural_model_type='sonata' invece di aggiungere un ramo per nome qui.
    if getattr(config, 'structural_model_type', None) == 'sonata':
        return _build_sonata_model(config)

    if getattr(config, 'structural_model_type', None) == 'ac38_tfoil':
        return _build_ac38_tfoil_model(config)

    if getattr(config, 'structural_model_type', None) == 'tfoil_simple':
        return _build_tfoil_simple_model(config)

    if getattr(config, 'structural_model_type', None) == 'ac37_tfoil':
        return _build_ac37_tfoil_model(config)

    if config.name in ('GOLAND', 'GOLAND_water'):
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
    elif config.name == 'Modelref2' or getattr(config, 'structural_model_type', None) == 'modelref2_tfoil':
        return _build_modelref2_tfoil_model(config)
    elif config.name in ('NACA0003', 'NACA0015_POM'):
        return _build_naca0003_model(config)
    elif _uses_custom_structural_model(config):
        return _build_structural_model_custom(config)
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
    print(f"Building {config.name} structural model from airfoil geometry")
 
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
    # 5.  EIxx CALIBRATION – match target dry 1st bending frequency
    f1_vac_target = float(getattr(config, 'f1_vac_target', 321.0))
    beta1L        = float(getattr(config, 'beta1L', 4.7300))
    omega1_target = 2.0 * np.pi * f1_vac_target
 
    EIxx_geom = EIxx               # geometry-only estimate (for reporting)
    EIxx      = mu * (omega1_target * beam_length**2 / beta1L**2)**2
    k_calib   = EIxx / EIxx_geom
 
    # Update mass rotational inertia for torsion consistently (J stays geometric)
    # i33 uses Ixx; now replace with calibrated Ixx equivalent for consistency:
    Ixx_cal = EIxx / E
    i33     = rho_s * Ixx_cal     # keep mass rotational inertia consistent
 
    calib_ref = getattr(config, 'f1_vac_reference', None) or config.name
    print(f"\n  ── EIxx frequency calibration ─────────────────────────────")
    print(f"    f₁,vac target  = {f1_vac_target:.1f} Hz  ({calib_ref})")
    print(f"    β₁L            = {beta1L}")
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
 
        # ── Mass (6×6) at EA ─────────────────────────────────────────────────
        M_i = np.zeros((6, 6), float)
        _set_section_mass_at_reference(
            M_i, mu, e_x, i3=i11, i4=i22, i5=i33, config=config,
        )
        M_dict[f"e{idx}"] = M_i
 
    # =========================================================================
    # 7.  ASSEMBLE BEAM MODEL
    # =========================================================================
    beam_model = create_beam_model(
        K_dict, M_dict,
        beam_length, n_elements,
        **beam_geometry_from_config(config),
        agard_theory=True,
        flutter_benchmark=False,
        sonata=False,
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
    
    print(f"\n{config.name} beam built: {n_elements} elements  span={beam_length:.3f} m")
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
                               config.output_dir)
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


def _build_modelref2_tfoil_model(config):
    """
    Strut of Reference 2 with inverted-T pod and foils (Besch & Liu 1973, Fig. 21–22).

    Assembly:
      1. Swept cantilever strut (Table 2 section properties).
      2. Two horizontal foil beams (±X from the strut tip) with distributed foil mass.
      3. Punctual pod mass at the strut tip node (``config.tip_mass`` / ``tip_mass_inertia``).

    Hydrodynamic added mass on pod/foils is handled separately in Capytaine (superposition).
    """
    from FEA.fea_utl.multibody_assembly import assemble_multibody, orient_element_section_from_local

    print(f"Building Modelref2 inverted-T structural model: {config.name}")

    K = np.zeros((6, 6), float)
    K[0, 0] = config.GAx
    K[1, 1] = config.EA
    K[2, 2] = config.GAz
    K[3, 3] = config.EIxx
    K[4, 4] = config.GJ
    K[5, 5] = config.EIzz

    xcm = config.xcm_factor * (config.chord / np.cos(np.deg2rad(config.sweep_angle_deg)))
    xsc = config.xea_factor * (config.chord / np.cos(np.deg2rad(config.sweep_angle_deg)))
    e = xcm - xsc
    on_sc = beam_on_shear_center(config)
    geo = beam_geometry_from_config(config)
    print(
        f"  Strut: L={config.beam_length:.4f} m, c={config.chord:.4f} m, "
        f"Λ={config.sweep_angle_deg:.1f}°, μ={config.mu:.4f} kg/m"
    )
    print(
        f"  xsc={xsc:.6f} m ({config.xea_factor:.3f}·c); "
        f"xcm={xcm:.6f} m ({config.xcm_factor:.3f}·c); e={e:.6f} m"
    )

    i11 = config.I_alpha / config.beam_length
    i22 = config.k_i22 * i11
    i33 = config.k_i33 * i11
    M = np.zeros((6, 6), float)
    _set_section_mass_at_reference(
        M, config.mu, e, i3=i22, i4=i11, i5=i33, config=config,
    )

    beam_strut = create_beam_model(
        K,
        M,
        config.beam_length,
        config.n_elements,
        **geo,
        cg_offset_e=e,
        beam_on_shear_center=on_sc,
        section_axes_along_global_span=bool(
            getattr(config, "section_axes_along_global_span", False)
        ),
        agard_theory=False,
        flutter_benchmark=True,
        sonata=False,
    )
    alpha_deg = float(getattr(config, "alpha_deg", 0.0))
    beam_strut = rotate_beam_model_y(beam_strut, [alpha_deg])

    sm_strut = {
        "nodes": [{**n, "submodel": "strut_ref2"} for n in beam_strut["nodes"]],
        "elements": beam_strut["elements"],
    }
    tip_pos = np.array(sm_strut["nodes"][-1]["position"], dtype=float)
    print(f"  Strut tip (foil junction): {tip_pos}")

    # ── Horizontal foil beams (±X, chord in Z) ───────────────────────────────
    L_foil = float(getattr(config, "foil_semi_span", 0.0))
    n_foil_el = int(getattr(config, "n_elements_foil", 8))
    mu_foil = float(getattr(config, "mu_foil", 0.0))
    foil_chord = float(getattr(config, "foil_root_chord", config.chord))
    ei_scale = float(getattr(config, "foil_EI_scale", 1.0))

    if bool(getattr(config, "foil_reuse_strut_section", False)):
        # Literal reuse of the strut's own K/M section (same EIxx, GJ, mu,
        # I_alpha, xea/xcm) on the foil beams -- foil_root_chord/foil_EI_scale/
        # mu_foil above are not used for stiffness/mass in this branch (kept
        # only for the aerodynamic planform / documentation).
        K_foil = K
        M_foil = M
        print("  Foils: reusing strut section (K, M) verbatim (foil_reuse_strut_section=True)")
    else:
        foil_EA = 70e9 * foil_chord * 0.15 * foil_chord * ei_scale
        foil_EI = foil_EA * foil_chord**2 / 12.0
        foil_GJ = foil_EI

        K_foil = np.zeros((6, 6), float)
        K_foil[0, 0] = foil_EA * 1e3
        K_foil[1, 1] = foil_EA
        K_foil[2, 2] = foil_EA * 1e3
        K_foil[3, 3] = foil_EI
        K_foil[4, 4] = foil_GJ
        K_foil[5, 5] = foil_EI

        i_foil = mu_foil * foil_chord**2 / 12.0
        M_foil = np.zeros((6, 6), float)
        M_foil[0, 0] = mu_foil
        M_foil[1, 1] = mu_foil
        M_foil[2, 2] = mu_foil
        M_foil[3, 3] = i_foil / 2.0
        M_foil[4, 4] = i_foil
        M_foil[5, 5] = i_foil / 2.0

    # Horizontal foils in the reference frame: span along ±Z (→ lateral ±Y after the
    # −90° structural dihedral about +X), chord along +X (streamwise), thin along Y.
    # Leading-edge / elastic-axis sweep per Fig. 22 (constant chord → EA sweep = LE
    # sweep): as the foil runs outboard (|z|↑) the elastic axis moves aft (+X).
    foil_sweep = float(getattr(config, "foil_sweep_deg", 0.0))
    tan_sweep = np.tan(np.deg2rad(foil_sweep))
    chord_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    thick_dir = np.array([0.0, 1.0, 0.0], dtype=float)

    def _foil_submodel(sign: float, name: str):
        nodes = []
        elements = []
        for i in range(n_foil_el + 1):
            s = (i / n_foil_el) * L_foil            # lateral (span) coordinate ≥ 0
            pos = tip_pos + np.array([s * tan_sweep, 0.0, sign * s], dtype=float)
            nodes.append({"position": pos.tolist(), "index": i, "submodel": name})
        for i in range(n_foil_el):
            elem = {
                "nodes": [i, i + 1],
                "length": L_foil / n_foil_el,
            }
            orient_element_section_from_local(
                elem,
                nodes[i]["position"],
                nodes[i + 1]["position"],
                K_foil,
                M_foil,
                chord_direction_global=chord_dir,
                thickness_direction_global=thick_dir,
            )
            elements.append(elem)
        return {"nodes": nodes, "elements": elements}

    # Incremental-validation switches (Besch & Liu Fig 15a/b/c): build strut only,
    # strut+pod, or the full T-foil from the same code path. Both default True.
    include_foils = bool(getattr(config, "modelref2_include_foils", True))
    include_pod = bool(getattr(config, "modelref2_include_pod", True))

    if include_foils:
        sm_foil_port = _foil_submodel(+1.0, "foil_port")
        sm_foil_stbd = _foil_submodel(-1.0, "foil_starboard")
        submodels = [sm_strut, sm_foil_port, sm_foil_stbd]
        sub_names = ["strut_ref2", "foil_port", "foil_starboard"]
    else:
        submodels = [sm_strut]
        sub_names = ["strut_ref2"]
        print("  [modelref2_include_foils=False] strut-only structure (no foils)")

    beam_model = assemble_multibody(submodels, tol=1e-7)
    beam_model["is_multibody"] = True
    beam_model["sub_beam_names"] = sub_names
    beam_model["K_section"] = K
    beam_model["M_section"] = M
    beam_model["geometry"] = dict(beam_strut.get("geometry", {}))
    beam_model["geometry"]["beam_on_shear_center"] = on_sc
    beam_model["geometry"]["xsc_factor"] = float(config.xea_factor)
    beam_model["geometry"]["xcm_factor"] = float(config.xcm_factor)

    tip_idx = None
    for idx, node in enumerate(beam_model["nodes"]):
        if np.linalg.norm(np.asarray(node["position"], float) - tip_pos) < 1e-9:
            tip_idx = idx
            break
    if tip_idx is None:
        tip_idx = len(beam_model["nodes"]) - 1

    tip_mass = float(getattr(config, "tip_mass", 0.0) or 0.0)
    if not include_pod:
        tip_mass = 0.0
        print("  [modelref2_include_pod=False] no tip pod point mass")
    if tip_mass > 0.0:
        inertia_cfg = getattr(config, "tip_mass_inertia", None)
        inertia = (
            np.diag(np.asarray(inertia_cfg, dtype=float)) if inertia_cfg is not None else None
        )
        offset_cfg = getattr(config, "tip_mass_offset", None)
        offset = (
            tuple(float(v) for v in offset_cfg) if offset_cfg is not None else (0.0, 0.0, 0.0)
        )
        beam_model["point_masses"] = [
            {
                "node_index": tip_idx,
                "mass": tip_mass,
                "inertia": inertia,
                "offset": offset,
            }
        ]
        print(
            f"  Pod point mass: {tip_mass:.4f} kg at node {tip_idx} "
            f"(CG offset={offset} m)"
        )

    print(
        f"Modelref2 T-foil assembled: {len(beam_model['nodes'])} nodes, "
        f"{len(beam_model['elements'])} elements "
        f"(strut {config.n_elements} + foils 2×{n_foil_el})"
    )
    return beam_model


def _build_structural_model_custom(config):
    """
    General uniform-span structural model from config section properties.

    Builds 6×6 ``K`` and ``M`` with DOF order ``[u,v,w,θx,θy,θz]`` (u chord, v span, w thick).
    Per-element ``T6`` maps them to global (swept elastic axis by default, or fixed +Y when
    ``section_axes_along_global_span``). Planform sweep / EA offset via ``beam_geometry_from_config``;
    optional angle of attack about +Y. Structural dihedral about +X is applied later in ``main.py``.

    Used for Abramson-style hydrofoil / wing-tunnel cases (``Model1``, ``ABRAMSON1965``,
    ``Model2-bluntbased``, …) when ``structural_model_type == 'custom'`` or the case name
    is listed in ``_CUSTOM_STRUCTURAL_MODEL_NAMES``.
    """
    print(f"Building custom structural model: {config.name}")

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

    # Accoppiamento flesso-torsionale (composite bend-twist tailoring).
    # psi = K_bt / sqrt(EIxx*GJ), adimensionale; |psi| < 1 e' richiesto perche'
    # il blocco 2x2 resti definito positivo. E' l'equivalente, in un modello a
    # trave dritta, della freccia: psi > 0 da' WASH-IN (flette su -> naso su,
    # destabilizzante, come freccia in avanti), psi < 0 da' WASH-OUT.
    # Si attiva solo se il config lo espone: gli altri casi restano intatti.
    _psi = float(getattr(config, "bend_twist_psi", 0.0) or 0.0)
    if abs(_psi) > 1e-12:
        if abs(_psi) >= 1.0:
            raise ValueError(
                f"bend_twist_psi={_psi} non ammissibile: serve |psi|<1 perche' "
                "il blocco flessione-torsione resti definito positivo"
            )
        _kbt = _psi * np.sqrt(config.EIxx * config.GJ)
        K[3, 4] = _kbt
        K[4, 3] = _kbt
        print(f"  bend-twist coupling: psi={_psi:+.3f} -> K_bt={_kbt:.4g} N m^2 "
              f"({'WASH-IN (tipo freccia avanti)' if _psi > 0 else 'WASH-OUT'})")

    xcm = config.xcm_factor * (config.chord / np.cos(np.deg2rad(config.sweep_angle_deg)))
    xsc = config.xea_factor * (config.chord / np.cos(np.deg2rad(config.sweep_angle_deg)))
    e = (xcm - xsc)  # CG offset from shear center along local chord (u); positive if CG is aft of SC
    on_sc = beam_on_shear_center(config)
    axis_label = "shear center" if on_sc else "beam reference"
    geo = beam_geometry_from_config(config)
    print(
        f"  Beam nodes on {axis_label} "
        f"(root x={geo['beam_axis_x_offset']:.6f} m); "
        f"xsc={xsc:.6f} m from LE ({config.xea_factor:.3f}·c); "
        f"xcm={xcm:.6f} m from LE ({config.xcm_factor:.3f}·c); "
        f"CG offset e={e:.6f} m from SC in section M only"
    )
    if xea_decoupled_from_aerogrid(config):
        print(
            f"  decouple_xea_from_aerogrid: aerogrid fixed at aerogrid_xea_factor="
            f"{aerogrid_xea_factor(config):.3f}; "
            f"beam chordwise shift Δx={structural_xea_chordwise_offset(config):.6f} m"
        )

    M = np.zeros((6, 6), float)

    if config.radius_gyration is not None:
        r2 = config.radius_gyration
        b = config.chord / 2
        r2_dim = r2 * (b**2)

        i11 = config.mu * r2_dim
        i22 = config.k_i22 * i11
        i33 = config.k_i33 * i11
        print(f"  Rotational inertia from r_α²={r2:.4f} (semichords): i11={i11:.4e} kg·m")

    elif config.I_alpha is not None:
        i11 = config.I_alpha / config.beam_length
        i22 = config.k_i22 * i11
        i33 = config.k_i33 * i11
        print(
            f"  Rotational inertia from I_α={config.I_alpha} "
            f"i11={i11:.4e} kg·m"
        )
    else:
        raise ValueError(
            f"{config.name}: set either radius_gyration or I_alpha for section rotational inertia"
        )

    _set_section_mass_at_reference(
        M, config.mu, e, i3=i22, i4=i11, i5=i33, config=config,
    )

    # Reference layout at build; structural dihedral about +X is applied in main.py.
    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements,
        **geo,
        cg_offset_e=e,
        beam_on_shear_center=on_sc,
        section_axes_along_global_span=bool(
            getattr(config, "section_axes_along_global_span", False)
        ),
        agard_theory=False,
        flutter_benchmark=True,
        sonata=False,
    )

    alpha_deg = float(getattr(config, "alpha_deg", 0.0))
    beam_model = rotate_beam_model_y(beam_model, [alpha_deg])

    # Section 6×6 and mesh stay in the reference frame (beam along +Y after create);
    # structural dihedral about +X is applied after global K/M and aerogrid are built
    # (see post_dihedral_utils.apply_structural_dihedral_about_x in main.py).
    beam_model["K_section"] = K
    beam_model["M_section"] = M
    beam_model["geometry"]["beam_on_shear_center"] = on_sc
    beam_model["geometry"]["xsc_factor"] = float(config.xea_factor)
    beam_model["geometry"]["xcm_factor"] = float(config.xcm_factor)
    beam_model["geometry"]["decouple_xea_from_aerogrid"] = xea_decoupled_from_aerogrid(config)
    if xea_decoupled_from_aerogrid(config):
        beam_model["geometry"]["aerogrid_xea_factor"] = float(aerogrid_xea_factor(config))
    if abs(alpha_deg) > 1e-12:
        print(f"  {config.name}: applied alpha {alpha_deg:.4f} deg about +Y on beam model (reference build).")

    # ── Optional concentrated tip mass (config.tip_mass [kg]) ────────────────
    # Registered in beam_model['point_masses']; structural_analysis adds it to
    # the free-end node in M_global. Optional:
    #   config.tip_mass_inertia = (Ixx, Iyy, Izz) [kg·m²] about the tip-mass CG,
    #       build reference frame (X chord +aft, Y span, Z thickness);
    #   config.tip_mass_offset  = (rx, ry, rz) [m] node → CG (e.g. pod CG aft
    #       of the elastic axis → positive rx). Unbalance coupling and
    #       parallel-axis terms are assembled in structural_analysis.
    tip_mass = float(getattr(config, 'tip_mass', 0.0) or 0.0)
    if tip_mass > 0.0:
        tip_idx = len(beam_model['nodes']) - 1
        inertia_cfg = getattr(config, 'tip_mass_inertia', None)
        inertia = (np.diag(np.asarray(inertia_cfg, dtype=float))
                   if inertia_cfg is not None else None)
        offset_cfg = getattr(config, 'tip_mass_offset', None)
        offset = (tuple(float(v) for v in offset_cfg)
                  if offset_cfg is not None else (0.0, 0.0, 0.0))
        beam_model['point_masses'] = [{
            'node_index': tip_idx,
            'mass':       tip_mass,
            'inertia':    inertia,
            'offset':     offset,
        }]
        print(f"  Tip point mass: {tip_mass:.3f} kg at free-end node {tip_idx} "
              f"(pos={beam_model['nodes'][tip_idx]['position']}), CG offset={offset} m"
              + (f", inertia diag (about CG)={inertia_cfg}" if inertia_cfg is not None else ""))
    
    # Optionally save the section stiffness and mass matrices to CSV files.
    # Controlled by config.save_matrices (bool). If True, matrices are written
    # to config.matrices_dir if provided, otherwise to ./matrices/.
    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               config.output_dir)
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

    # Accoppiamento flesso-torsionale (composite bend-twist tailoring):
    # psi = K_bt/sqrt(EIxx*GJ), |psi|<1 per definita positivita'. Equivalente,
    # su trave dritta, della freccia: psi>0 = WASH-IN (flette su -> naso su,
    # destabilizzante, come freccia in avanti). Attivo solo se il config lo espone.
    _psi = float(getattr(config, "bend_twist_psi", 0.0) or 0.0)
    if abs(_psi) > 1e-12:
        if abs(_psi) >= 1.0:
            raise ValueError(f"bend_twist_psi={_psi}: serve |psi|<1")
        _kbt = _psi * np.sqrt(config.EIxx * config.GJ)
        K[3, 4] = _kbt
        K[4, 3] = _kbt
        print(f"  bend-twist: psi={_psi:+.3f} -> K_bt={_kbt:.4g} "
              f"({'WASH-IN' if _psi > 0 else 'WASH-OUT'})")

    xcm = config.xcm_factor * config.chord
    xea = config.xea_factor * config.chord
    e = (xcm - xea)  # CG offset from shear center along local chord (+x = aft of SC)
    on_sc = beam_on_shear_center(config)
    axis_label = "shear center" if on_sc else "beam reference"
    print(
        f"  Beam nodes on {axis_label} (root at origin); "
        f"xsc={xea:.6f} m from LE ({config.xea_factor:.3f}·c); "
        f"xcm={xcm:.6f} m from LE ({config.xcm_factor:.3f}·c); "
        f"CG offset e={e:.6f} m from SC in section M only"
    )

    M = np.zeros((6, 6), float)
    _set_section_mass_at_reference(
        M,
        config.mu,
        e,
        i3=config.i22,
        i4=config.i11,
        i5=config.i33,
        config=config,
    )

    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements,
        **beam_geometry_from_config(config),
        cg_offset_e=e,
        beam_on_shear_center=on_sc,
        agard_theory=False,
        flutter_benchmark=True,
        sonata=False,
    )

    if bool(getattr(config, "structural_alpha_rotate", False)):
        beam_model = rotate_beam_model_y(beam_model, [config.alpha_deg])
    else:
        print(f"  Structural mesh unrotated (α={config.alpha_deg}° applied in aero only, SHARPy convention)")
    
    # Add matrices to the model for later use
    beam_model['K_section'] = K
    beam_model['M_section'] = M
    beam_model["geometry"]["beam_on_shear_center"] = on_sc
    beam_model["geometry"]["xsc_factor"] = float(config.xea_factor)
    beam_model["geometry"]["xcm_factor"] = float(config.xcm_factor)
    
    # Optionally save the section stiffness and mass matrices to CSV files.
    # Controlled by config.save_matrices (bool). If True, matrices are written
    # to config.matrices_dir if provided, otherwise to ./matrices/.
    try:
        save_flag = bool(getattr(config, 'save_matrices', False))
    except Exception:
        save_flag = False

    if save_flag:
        matrices_dir = getattr(config, 'matrices_dir',
                               config.output_dir)
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

    M = np.zeros((6, 6), float)
    _set_section_mass_at_reference(
        M,
        config.mu,
        e,
        i3=config.i22,
        i4=config.i11,
        i5=config.i33,
        config=config,
    )

    beam_model = create_beam_model(
        K, M, config.beam_length, config.n_elements,
        **beam_geometry_from_config(config),
        cg_offset_e=e,
        agard_theory=False,
        flutter_benchmark=True,
        sonata=False,
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
                               config.output_dir)
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

    geo = beam_geometry_from_config(config)

    beam_model = create_beam_model(
        K, M,
        config.n_elements,
        sections_props_csv=sections_props_csv,
        beam_axis_x_offset=geo["beam_axis_x_offset"],
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
                               config.output_dir)
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
        _set_section_mass_at_reference(
            Mi,
            mu_i,
            e,
            i3=i11_el[i],
            i4=i22_el[i],
            i5=i33_el[i],
            config=config,
        )
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

        K_loc = K_dict_arm[f"e{i}"]
        M_loc = M_dict_arm[f"e{i}"]

        elem = {"nodes": [i, i + 1], "length": L_e}
        orient_element_section_from_local(elem, p1, p2, K_loc, M_loc)
        elements_arm.append(elem)
    
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

            link_elem = {
                'nodes': [anchor_idx, ghost_idx],
                'length': link_len,
                'rigid_link': True,
            }
            orient_element_section_from_local(
                link_elem, anchor_pos, cg_m, K_link_loc, M_link_loc
            )
            beam_model['elements'].append(link_elem)
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
                               config.output_dir)
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


def _build_tfoil_simple_model(config):
    """T-foil semplificato (examples/TFOIL_SIMPLE): timone verticale + due semiali.

    Geometria e sezioni da ``examples/TFOIL_SIMPLE/tfoil_simple_geom.py``. Frame di
    GINO come AC38_TFOIL prima del diedro: x a poppa (verso della corrente), y lungo
    il timone verso il basso (0 all'incastro, z = -90 mm), elevatore lungo z
    (semiala 'elevator_stbd' a z < 0). Asse della trave sull'asse elastico di ogni
    sezione; le semiali partono dalla PUNTA del timone (nodo condiviso, niente
    collegamento rigido: nel caso semplificato non c'e' bulbo).
    """
    import importlib.util as _ilu
    import numpy as _np
    from FEA.fea_utl.multibody_assembly import (assemble_multibody,
                                                orient_element_section_from_local)

    _p = _os_path_join(config.paths['FSI'], 'examples', 'TFOIL_SIMPLE', 'tfoil_simple_geom.py')
    _s = _ilu.spec_from_file_location('tfoil_simple_geom', _p)
    G = _ilu.module_from_spec(_s); _s.loader.exec_module(G)
    print(f"Building TFOIL_SIMPLE structural model: {G.tag()}")

    ne_r = int(getattr(config, 'n_elements', 52))
    ne_e = int(getattr(config, 'n_elements_elevator', 18))
    chord_dir = _np.array([1.0, 0.0, 0.0])

    def _branch(pos, chords, name):
        nodes = [{"position": p.tolist(), "index": i, "submodel": name} for i, p in enumerate(pos)]
        elements = []
        for i in range(len(pos) - 1):
            K, M = G.section(0.5 * (chords[i] + chords[i + 1]))
            el = {"nodes": [i, i + 1], "length": float(_np.linalg.norm(pos[i + 1] - pos[i]))}
            orient_element_section_from_local(el, pos[i], pos[i + 1], K, M,
                                              chord_direction_global=chord_dir)
            elements.append(el)
        return {"nodes": nodes, "elements": elements}

    yr = _np.linspace(0.0, G.L_RUDDER, ne_r + 1)
    pos_r = _np.column_stack([_np.zeros_like(yr), yr, _np.zeros_like(yr)])
    sm_r = _branch(pos_r, G.chord_rudder(-(G.Z_ROOT - yr)), "rudder")
    sm_r["nodes"][0]["clamped"] = True
    s = _np.linspace(0.0, G.ELEV_SEMISPAN, ne_e + 1)
    subs = [sm_r]
    edx = float(getattr(G, "EDX", 0.0))
    for sign, name in ((-1.0, "elevator_stbd"), (+1.0, "elevator_port")):
        pos = _np.column_stack([_np.full_like(s, edx), _np.full_like(s, G.L_RUDDER), sign * s])
        subs.append(_branch(pos, G.chord_elev(s), name))
    if edx:
        # collegamento RIGIDO punta del timone -> radice dell'elevatore (come AC38_TFOIL):
        # rigidezza isotropa 100 volte la piu' alta della radice, massa nulla
        K0r, _ = G.section(G.C_ROOT)
        tip = _np.array([0.0, G.L_RUDDER, 0.0]); root = _np.array([edx, G.L_RUDDER, 0.0])
        subs.append({"nodes": [{"position": tip.tolist(), "index": 0, "submodel": "link"},
                               {"position": root.tolist(), "index": 1, "submodel": "link"}],
                     "elements": [{"nodes": [0, 1], "length": abs(edx),
                                   "stiffness": 1e2 * float(_np.max(_np.diag(K0r))) * _np.eye(6),
                                   "mass": _np.zeros((6, 6))}]})
        print(f"  Elevatore spostato di {edx*1e3:+.1f} mm lungo x (collegamento rigido senza massa)")
    beam_model = assemble_multibody(subs, tol=1e-7)
    beam_model["is_multibody"] = True
    beam_model["sub_beam_names"] = ["rudder", "elevator_stbd", "elevator_port"] + (["link"] if edx else [])
    K0, M0 = G.section(G.C_ROOT)
    beam_model["K_section"] = K0
    beam_model["M_section"] = M0
    m_tot = sum(el["section_mass_local"][0, 0] * el["length"] for el in beam_model["elements"] if "section_mass_local" in el)
    print(f"  {len(beam_model['nodes'])} nodi, {len(beam_model['elements'])} elementi, massa {m_tot:.1f} kg")
    print("  " + G.summary().replace("\n", "\n  "))
    return beam_model


def _build_ac37_tfoil_model(config):
    """Timone AC37 (RE143 / RE202, examples/AC37_TFOIL): trave verticale + due semiali.

    Tabelle di trave ASWING (EI, GK, massa, centro di taglio) da
    ``examples/AC37_TFOIL/ac37_geom.py``. Frame come TFOIL_SIMPLE: x a poppa, y lungo il
    timone verso il basso (0 alla boccola inferiore), elevatore lungo z. Le travi seguono
    la linea dei centri di taglio; elevatore collegato alla punta del timone da un
    elemento rigido senza massa quando i due assi non coincidono.
    """
    import importlib.util as _ilu
    import numpy as _np
    from FEA.fea_utl.multibody_assembly import (assemble_multibody,
                                                orient_element_section_from_local)

    _p = _os_path_join(config.paths['FSI'], 'examples', 'AC37_TFOIL', 'ac37_geom.py')
    _s = _ilu.spec_from_file_location('ac37_geom', _p)
    G = _ilu.module_from_spec(_s); _s.loader.exec_module(G)
    print(f"Building AC37_TFOIL structural model: {G.tag()}")

    ne_r = int(getattr(config, 'n_elements', 56))
    ne_e = int(getattr(config, 'n_elements_elevator', 18))
    chord_dir = _np.array([1.0, 0.0, 0.0])
    L_r, L_e = G.Y_END - G.Y_TOP, G.ELEV_SEMISPAN

    def _branch(pos, props, L, name):
        nodes = [{"position": p.tolist(), "index": i, "submodel": name} for i, p in enumerate(pos)]
        elements = []
        for i in range(len(pos) - 1):
            K, M = G.section(props(i), L)
            el = {"nodes": [i, i + 1], "length": float(_np.linalg.norm(pos[i + 1] - pos[i]))}
            orient_element_section_from_local(el, pos[i], pos[i + 1], K, M,
                                              chord_direction_global=chord_dir)
            elements.append(el)
        return {"nodes": nodes, "elements": elements}

    pos_r = G.rudder_nodes(ne_r)
    sm_r = _branch(pos_r, lambda i: G.rudder_props(0.5 * (pos_r[i, 1] + pos_r[i + 1, 1]), (pos_r[i, 1], pos_r[i + 1, 1])),
                   L_r, "rudder")
    sm_r["nodes"][0]["clamped"] = G.SUPPORT == "clamp"
    subs = [sm_r]
    for sign, name in ((-1.0, "elevator_stbd"), (+1.0, "elevator_port")):
        pos, Y = G.elev_nodes(ne_e, sign)
        subs.append(_branch(pos, lambda i, Y=Y: G.elev_props(0.5 * (Y[i] + Y[i + 1]), (Y[i], Y[i + 1])), L_e, name))
    if G.LINK:
        # collegamento RIGIDO punta del timone -> radice dell'elevatore (come AC38_TFOIL)
        K0r, _ = G.section(G.rudder_props(0.5), L_r)
        tip = pos_r[-1]; root = G.elev_nodes(ne_e, 1.0)[0][0]
        subs.append({"nodes": [{"position": tip.tolist(), "index": 0, "submodel": "link"},
                               {"position": root.tolist(), "index": 1, "submodel": "link"}],
                     "elements": [{"nodes": [0, 1], "length": float(_np.linalg.norm(root - tip)),
                                   "stiffness": 1e2 * float(_np.max(_np.diag(K0r))) * _np.eye(6),
                                   "mass": _np.zeros((6, 6))}]})
        print(f"  Collegamento rigido timone -> elevatore: dx = {(root[0]-tip[0])*1e3:+.1f} mm")
    beam_model = assemble_multibody(subs, tol=1e-7)
    beam_model["is_multibody"] = True
    beam_model["sub_beam_names"] = ["rudder", "elevator_stbd", "elevator_port"] + (["link"] if G.LINK else [])
    K0, M0 = G.section(G.rudder_props(1.0), L_r)
    beam_model["K_section"] = K0
    beam_model["M_section"] = M0
    m_tot = sum(el["section_mass_local"][0, 0] * el["length"] for el in beam_model["elements"] if "section_mass_local" in el)
    print(f"  {len(beam_model['nodes'])} nodi, {len(beam_model['elements'])} elementi, massa {m_tot:.2f} kg")
    print("  " + G.summary().replace("\n", "\n  "))
    return beam_model


def _os_path_join(*a):
    import os as _os
    return _os.path.join(*a)


def _build_ac38_tfoil_model(config):
    """Timone AC38 a T: trave verticale + le due semiali dell'elevatore.

    TRE SOTTOMODELLI e un collegamento.

      1. il TIMONE, dalla stessa csv_export del caso a sola verticale: nodi
         lungo +Y (nel frame prima del diedro), sull'asse di rotazione.
      2/3. le due SEMIALI dell'elevatore, da ``out/beam_ELEV/elevator_beam.npz``,
         gia' nel frame globale. La semiala sinistra e' lo specchio z -> -z di
         quella destra, con i momenti che prendono il segno in piu' dovuto allo
         specchio (vedi beam/export_elevator_beam.py).
      4. un elemento RIGIDO che collega la punta della lama alla radice
         dell'elevatore.

    PERCHE' UN ELEMENTO E NON UNA SALDATURA DI NODI. I due punti non coincidono:
    la punta della lama sta sull'asse di rotazione, la radice dell'elevatore
    102.7 mm a poppa e 38.5 mm piu' in basso. Quell'offset e' un braccio, e un
    braccio trasmette momento: ignorarlo e saldare i nodi toglierebbe
    l'accoppiamento fra la flessione dell'elevatore e la TORSIONE del timone,
    che e' il meccanismo da cui dipende il flutter di un timone a T.

    Il collegamento e' rigido perche' cosi' lo prescrive RevD (nota 13), ed e'
    fisicamente il bulbo d'acciaio che fascia la punta della lama con 27 viti.
    Si realizza con un elemento mille volte piu' rigido del membro piu' rigido
    che collega e con massa NULLA: la massa del bulbo c'e' gia' nelle sezioni
    dell'elevatore vicino a y = 0, e contarla di nuovo la raddoppierebbe.
    """
    import os as _os
    import numpy as _np
    from FEA.fea_utl.multibody_assembly import assemble_multibody, T6_from_beam_and_chord

    print(f"Building AC38 T-foil structural model: {config.name}")

    # ── 1. verticale ─────────────────────────────────────────────────────────
    beam_rudder = _build_sonata_model(config)
    sm_rudder = {
        "nodes": [{**n, "submodel": "rudder"} for n in beam_rudder["nodes"]],
        "elements": beam_rudder["elements"],
    }
    tip = _np.asarray(sm_rudder["nodes"][-1]["position"], float)
    print(f"  Timone: {len(sm_rudder['nodes'])} nodi, punta della lama {tip}")

    # ── 2/3. elevatore ───────────────────────────────────────────────────────
    npz = getattr(config, "elevator_beam_npz", None)
    if not npz:
        raise ValueError("config.elevator_beam_npz non impostato: serve la tabella "
                         "prodotta da beam/export_elevator_beam.py")
    d = _np.load(npz)
    y = _np.asarray(d["y_mm"], float).copy()
    P = _np.asarray(d["pos"], float).copy()
    Kg = _np.asarray(d["K"], float)
    Mg = _np.asarray(d["M"], float)
    MIR6 = _np.asarray(d["mir6"], float)
    S = _np.diag([1.0, 1.0, -1.0])

    # la prima stazione sta a y = 0.5 mm: la si porta esattamente a y = 0 cosi'
    # le radici delle due semiali coincidono e il multibody le salda fra loro
    P[0, 2] = 0.0
    # AC38_ELEV_DX [mm]: sensitivita' allo stagger, elevatore traslato verso prua
    # (x negativo nella terna di GINO) di questa quantita'
    _dx = float(_os.environ.get("AC38_ELEV_DX", "0") or 0.0)
    if _dx:
        P[:, 0] -= _dx * 1e-3
        print(f"  Elevatore traslato di {_dx:.1f} mm verso prua (AC38_ELEV_DX)")
    # AC38_ELEV_SHIFT=<csv y_mm,dx_mm>: ogni sezione traslata rigidamente verso prua di dx(y) (asse elastico
    # dritto, 30_asse_elastico_dritto). K e M sono riferite al centro di tensione, che trasla con la sezione:
    # cambia solo la posizione dei nodi
    _sh = _os.environ.get("AC38_ELEV_SHIFT")
    if _sh:
        _t = _np.loadtxt(_sh, delimiter=",", comments="#", ndmin=2)
        P[:, 0] -= _np.interp(y, _t[:, 0], _t[:, 1]) * 1e-3
        print(f"  Sezioni dell'elevatore traslate da {_sh} (dx {_t[:, 1].min():.1f}..{_t[:, 1].max():.1f} mm)")
    n_el_half = int(getattr(config, "n_elements_elevator", len(y) - 1))
    if n_el_half != len(y) - 1:
        # ricampionamento uniforme sullo span, se richiesto
        ys = _np.linspace(y[0], y[-1], n_el_half + 1)
        P = _np.column_stack([_np.interp(ys, y, P[:, k]) for k in range(3)])
        Kg = _np.array([[[_np.interp(t, y, Kg[:, i, j]) for j in range(6)]
                         for i in range(6)] for t in ys])
        Mg = _np.array([[[_np.interp(t, y, Mg[:, i, j]) for j in range(6)]
                         for i in range(6)] for t in ys])
        y = ys
        P[0, 2] = 0.0

    def _half(side, name):
        """side=+1 dritta (z_G negativo), side=-1 sinistra."""
        pos = P.copy()
        K_, M_ = Kg, Mg
        if side < 0:
            pos[:, 2] = -pos[:, 2]
            K_ = _np.einsum('ij,njk,lk->nil', MIR6, Kg, MIR6)
            M_ = _np.einsum('ij,njk,lk->nil', MIR6, Mg, MIR6)
        nodes, elements = [], []
        for i, p in enumerate(pos):
            nodes.append({"position": p.tolist(), "index": i, "submodel": name})
        for i in range(len(pos) - 1):
            d_ = pos[i + 1] - pos[i]
            L = float(_np.linalg.norm(d_))
            # La terna dell'elemento va FISSATA alla corda. Senza, GINO la
            # deduce dalla sola direzione dell'elemento e sceglie una
            # perpendicolare qualunque: su un'ala a freccia quella scelta ruota
            # la sezione nel suo piano e scambia EI_flap con EI_chord, che qui
            # differiscono di un fattore 40.
            elements.append({
                "nodes": [i, i + 1], "length": L,
                "stiffness": 0.5 * (K_[i] + K_[i + 1]),
                "mass": 0.5 * (M_[i] + M_[i + 1]),
                "T6": T6_from_beam_and_chord(d_ / L, _np.array([1.0, 0.0, 0.0])),
            })
        return {"nodes": nodes, "elements": elements}

    sm_stbd = _half(+1, "elevator_stbd")
    sm_port = _half(-1, "elevator_port")
    root = _np.asarray(sm_stbd["nodes"][0]["position"], float)
    print(f"  Elevatore: 2 x {len(y)} nodi, radice {root}, "
          f"semiapertura {abs(P[-1, 2]):.4f} m")

    # ── 4. collegamento rigido ───────────────────────────────────────────────
    off = root - tip
    # Rigidezza ISOTROPA, non la diagonale della radice scalata: un elemento
    # rigido non deve avere direzioni preferite, e con una diagonale anisotropa
    # il risultato dipenderebbe da come l'elemento e' orientato nello spazio.
    k_scale = float(getattr(config, "tfoil_link_stiffness_factor", 1.0e2))
    K_link = k_scale * float(_np.max(_np.diag(Kg[0]))) * _np.eye(6)
    sm_link = {
        "nodes": [{"position": tip.tolist(), "index": 0, "submodel": "link"},
                  {"position": root.tolist(), "index": 1, "submodel": "link"}],
        "elements": [{"nodes": [0, 1], "length": float(_np.linalg.norm(off)),
                      "stiffness": K_link, "mass": _np.zeros((6, 6))}],
    }
    print(f"  Collegamento rigido: offset ({off[0]*1e3:+.1f}, {off[1]*1e3:+.1f}, "
          f"{off[2]*1e3:+.1f}) mm, lunghezza {_np.linalg.norm(off)*1e3:.1f} mm, "
          f"rigidezza isotropa {k_scale:g}x la piu' alta della radice, massa nulla")

    beam_model = assemble_multibody([sm_rudder, sm_link, sm_stbd, sm_port], tol=1e-7)
    beam_model["is_multibody"] = True
    beam_model["sub_beam_names"] = ["rudder", "link", "elevator_stbd", "elevator_port"]
    beam_model["K_section"] = beam_rudder.get("K_section")
    beam_model["M_section"] = beam_rudder.get("M_section")
    print(f"  Assemblato: {len(beam_model['nodes'])} nodi, "
          f"{len(beam_model['elements'])} elementi")
    return beam_model
