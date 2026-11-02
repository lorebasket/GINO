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


def _build_private_model(config):
    """Casi non pubblicati: private_models.py e' opzionale e resta fuori dal repository."""
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'private_models.py')
    if not os.path.isfile(_p):
        return None
    _s = _ilu.spec_from_file_location('private_models', _p)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m.build(config, sys.modules[__name__])


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

    if getattr(config, 'structural_model_type', None) == 'tfoil_simple':
        return _build_tfoil_simple_model(config)

    _private = _build_private_model(config)
    if _private is not None:
        return _private

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


def _build_tfoil_simple_model(config):
    """T-foil semplificato (examples/TFOIL_SIMPLE): timone verticale + due semiali.

    Geometria e sezioni da ``examples/TFOIL_SIMPLE/tfoil_simple_geom.py``. Frame di
    GINO prima del diedro: x a poppa (verso della corrente), y lungo
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
        # collegamento RIGIDO punta del timone -> radice dell'elevatore:
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


def _os_path_join(*a):
    import os as _os
    return _os.path.join(*a)
