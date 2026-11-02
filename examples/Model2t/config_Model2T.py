import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    fluid            = 'water'
    attack_angle_deg = 0.0

    # ── Discretization ───────────────────────────────────────────────────
    n_elements=200
    nspan=24
    nchord=12

    qjj_dir = os.path.join(QJJ_path, f"Model2t_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    vjj_dir = os.path.join(QJJ_path, f"vjj_Model2t_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    # ── Model 2T pod, configuration G (Besch & Liu 1973, Report 4012) ────────
    # Fig. 3: pod weight 38.4 lb, CG ≈ 7.8 in aft of the strut elastic axis.
    # Fig. 20 (config G): 2.0 in diameter; 12 in nose + 25.7 in body = 37.7 in
    # TOTAL length, welded at the tip with the nose 6 in ahead of the strut LE.
    # Inertias not given in the paper → recomputed as a solid prolate spheroid
    # of total mass m_pod (authors approximate the pod as a prolate spheroid):
    #   I_axial = (2/5) m b²           about the pod long axis (chord, X)
    #   I_trans = (1/5) m (a² + b²)    about transverse axes through the CG
    # Parallel-axis (m·d² ≈ 0.684 kg·m² → I_EA ≈ 1.057 kg·m² about the EA) and
    # bending–torsion unbalance coupling are assembled by structural_analysis
    # from tip_mass_offset.
    pod_mass   = 38.4 * 0.45359237   # [kg]  ≈ 17.42
    pod_length = 25.7 * 0.0254       # [m]   BODY-only length (Fig. 20); config G total is 37.7 in.
                                     #       Kept at 25.7 for the structural spheroid inertia — changing
                                     #       it alters dry modes and the GJ/EI calibrations.
    pod_diam   = 2.0  * 0.0254       # [m]   pod diameter
    pod_cg_aft = 7.8  * 0.0254       # [m]   pod CG aft of the elastic axis
    a_pod = pod_length / 2.0         # spheroid semi-major axis
    b_pod = pod_diam   / 2.0         # spheroid semi-minor axis
    I_pod_axial = (2.0 / 5.0) * pod_mass * b_pod**2                 # ≈ 0.0045 kg·m²
    I_pod_trans = (1.0 / 5.0) * pod_mass * (a_pod**2 + b_pod**2)    # ≈ 0.3734 kg·m²

    ort_beam_length = 49.693 * 0.0254  # [m]  Table 1 L=49.693 in (submerged span along swept EA = 48.0 in vertical / cos15deg) = 1.262 m
    ort_chord = 5.796 * 2 * 0.0254       # [m]  Table 1 chord = 11.5748 inches = 0.294 m

    wetted_distance_from_root = 10 * 0.0254  # [m]  Table 1: 10 in from root to the start of the pod (submerged span along swept EA = 9.66 in vertical / cos15deg) = 0.254 m
    sweep_angle_deg = 15.0
    xea_factor = 0.69   # shear center at 69 % chord from LE (a=+0.38)

    return AnalysisConfig(
        name='Model2t',
        blade_name='Model2t',
        fluid=fluid,

        n_elements=n_elements,

        # ── Units convertion ──────────────────────────────────────────────
        # 1 inch = 0,0254 m
        # 1 lb-in^2 = 2.87e-3 N-m^2
        # 1 lb*sec^2/in^2 = 6894.75 N-s^2/m^2 = 6894.75 kg/m
        # 1 lbf·in/s² = 0.112985 N * m * s^2

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=ort_beam_length,## * np.cos(np.deg2rad(15)),  # Table 1 L=49.693 in (submerged span along swept EA = 48.0 in vertical / cos15deg) = 1.262 m
        chord=ort_chord, #0.294,#/np.cos(np.deg2rad(15)), # 11.5748 inches = 0.294 m
        chord_root=ort_chord,#/np.cos(np.deg2rad(15)),
        chord_tip=ort_chord,#/np.cos(np.deg2rad(15)),
        thickness_factor=0.033,
        chordwise_strip_y=ort_chord,#/np.cos(np.deg2rad(15)),
        
        # Post-build rotation about +X (deg); see post_dihedral_utils.
        # dihedral_rotate_beam / dihedral_rotate_aerogrid: only trave, only DLM grid, or both (default).
        dihedral_angle=-90,  # vertically inside the test chamber
        dihedral_rotate_beam=True,
        dihedral_rotate_aerogrid=True,
        beam_span_axis=2,  # 2 = Z axis when dihedral_angle = -90 deg
        sweep_angle_deg=sweep_angle_deg,  # 15 deg sweep of the EA (chordwise) about the span axis (Z)
        section_axes_along_global_span=True, # Section K/M in global frame (v along +Y), not rotated with swept SC axis.

        # ── Material & mass ───────────────────────────────────────────────
        rho_s=7000,
        mu=18.8, # 0.00273 lb*sec^2/in^2 = 0.00273 * 6894.75 kg/m = 18.8 kg/m
        v=0.3,
        radius_gyration=None,
        I_alpha=0.16812,  # moment of inertia about EA [kg·m²] from 1.488 [lbf·s²/in²],

        # ── Model 2T pod, config G: concentrated mass at the free end ───────
        # Build reference frame: X chord (+aft of EA), Y span, Z thickness.
        # CG is 7.8 in aft of the EA (Fig. 3) and at the pod-cylinder centre:
        # the tip node sits on the strut cap; in build coords +Y by b_pod from the
        # tip maps to −Z (down the span) after R_x(−90°), i.e. the circle centre.
        tip_mass=pod_mass,                              # [kg] 17.42 (38.4 lb)
        tip_mass_offset=(pod_cg_aft, b_pod, 0.0),
        tip_mass_inertia=(I_pod_axial,                  # θx: about pod long axis
                          I_pod_trans,                  # θy: torsion (span axis)
                          I_pod_trans),                 # θz: about vertical axis

        # ── Shear center / CG (chordwise fraction from LE) ───────────────
        # Beam nodes on SC; root at (0,0,0). xea/xcm used in M and PanelAero coupling.
        beam_on_shear_center=True,
        beam_axis_x_offset=0.0,
        # Theodorsen (Table 1): a=+0.38 -> EA at 0.5+0.38/2 = 0.690c from LE; x_alpha=-0.214 -> CG at 0.583c (CG fwd of EA, both aft of midchord per blunt-aft section).
        # (Previously 0.31/0.417 = mirror about midchord; corrected to paper convention 2026-06-24.)
        xea_factor=xea_factor,   # shear center at 69 % chord from LE (a=+0.38)
        xcm_factor=0.583,  # center of gravity at 58.3 % chord from LE (x_alpha=-0.214)

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=3710.91, # 1.293e6 lbf-in^2 = 3710.91 N-m^2
        GJ= 5940.9, # 2.070e6 lbf-in^2 = 5940.9 N-m^2
        k_i22=0.3,
        k_i33=0.7,
        EA=3710.91*1e3,
        EIzz=3710.91*1e4,
        GAx=5940.9*1e3,
        GAz=5940.9*1e3,

        # ── Modal / Rayleigh (3 coupled flutter branches; 4th dry mode kept for basis) ──
        num_modes_egv=4,
        num_modes_flutter_egv=3,
        rayleigh_target_zetas=(0.05, 0.05, 0.05),

        # ── RFA + strip is the Model2-bluntbased benchmark line ───────
        pk_method=True,
        aero_b2_removal=True, # if true, the b2 matrix is removed from the Re(Q), RFA hydroelastic correction without RFA fit :)
        
        roger_fit=False,
        RFA_PK_method=False,
        
        rfa_fit_b2=True, # if true, the b2 matrix is fitted using the modal analysis
        rfa_apply_b2_mass=False, # if true, the b2 matrix is applied to the mass matrix
        
        added_mass_strip_theory=False,
        added_mass_strip_torsion=False,        # add 2-D pitch added inertia rho*pi*b^4*(1/8+a^2) on torsion DOF
        added_mass_strip_torsion_scale=1.0,   # tuning multiplier on the torsional added inertia

        # Roger tracking: wider pool, drop lag-dominated roots, softer branch guard
        # (branch-guard / mac-reference / predict-and-select set in the lag-pole block below)
        rfa_pool_n_roots=8,
        rfa_pool_structural_weight_min=0.20,
        rfa_pool_omega_max_frac=2.5,  # 1B stiffens to ~2x its rest freq (paper Fig 11b); 1.20 cap dropped it at ~12 kn

        # ── Im(Q) scale (RFA fit unscaled; applied in C_eff and optionally lag forcing) ──
        aero_im_Q_scale=1.0,  # try 0.3–0.7 if aero destabilizes too early vs Abramson ~48 kn
        aero_im_Q_scale_lags=False,

        # ── Empirical fluid damping (Abramson Fig. 5: ~Δδ≈4 s⁻¹ above measured-Q curve) ──
        empirical_fluid_damping=False,
        empirical_fluid_model='abramson_delta',
        empirical_fluid_delta_add_s=5.0, # It is recommended to use a value of 5.0 s⁻¹ for the empirical fluid damping.

        # ── Analysis struct params ────────────────────────────────────────
        modal_dir = os.path.join(FSI_path, "output_data", "Model2t"),
        prefix = "Model2t_dry_egv",

        # ── Flutter sweep ─────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        # Skip V < 2 m/s (high-k UVLM wake-root band); 40 kn ≈ 20.58 m/s (Besch & Liu Fig. 11)
        V_list=np.linspace(2.0, 20.58, 80),

        # ── Roger RFA lag poles (3 tracked wet modes; V_guess ≈ 53 kn branch-2 crossing) ──
        rfa_roger_convention='paper_zurich',
        mac_matching=True,
        last_converged_mode_matching=True,
        rfa_mac_reference='previous_velocity',
        rfa_predict_and_select=False,  # PAS off: avoids branch swaps / zig-zags above ~35 kn
        rfa_k_fit_adaptive_band=False,
        rfa_weight_fit_k=False,  # True overfits high-k and destabilises B1 (see repo notes)
        rfa_w_mac=0.85,
        rfa_w_freq=0.45,
        rfa_branch_guard_enable=True,
        rfa_branch_reacquire_enable=True,
        rfa_branch_guard_max_struct_frac=2.5,
        rfa_branch_guard_max_rel_jump=0.30,
        rfa_branch_guard_min_mac_prev=0.78,
        rfa_branch_guard_min_sep_frac=0.07,
        rfa_branch_guard_check_neighbor_collapse=True,
        rfa_branch_reacquire_min_mac=0.72,
        rfa_branch_reacquire_min_partner_sep_frac=0.06,
        
        n_lag=4,
        blag=np.array([0.5, 1.0, 1.5, 2.0]),  # dimensionless β_l (paper)
        rfa_blag_dimensionless=False, # True if rfa_roger_convention: 'paper_zurich', False if 'panelaero_legacy'
        rfa_blag_ref_length='semichord',

        # ── Capytaine sweep (ω grid denser in 5–120 rad/s = Model 2T wet/flutter band) ──
        capytaine_results_dir = os.path.join(FSI_path, "FLUID", "capytaine", "Model2t", "results_modal_radiation"),
        mesh_path = os.path.join(FSI_path, "FLUID", "capytaine", "Model2t", "Model2t_mesh.vtu"),
        # Superposition workflow: strut and pod BEM meshes drawn separately in plot_aerobeam.
        capytaine_strut_mesh_path = os.path.join(
            FSI_path, "FLUID", "capytaine", "Model2t", "Model2t_mesh_strutonly.vtu",
        ),
        capytaine_pod_mesh_path = os.path.join(
            FSI_path, "FLUID", "capytaine", "Model2t", "Model2t_mesh_podonly.vtu",
        ),
        # Mesh-convergence study (01_discretization_convergence_analysis, 2026-07-10):
        # 40x40 strut = 0.95% max A-error vs 48x48 reference (30x30 was 1.55%).
        mesh_n_span = 40,
        mesh_n_chord = 40,
        # ── Tip pod/bulb hydrodynamic added mass (Besch & Liu Fig. 20, config G) ──
        # The 2-in-diameter pod welded to the strut tip is added to the Capytaine
        # BEM mesh as a capped cylinder (axis streamwise) so its (large, torsion-
        # dominant) added mass is captured. Positioned per Fig. 20: nose 6 in ahead
        # of the tip leading edge, hung just below the strut tip (top at the tip).
        capytaine_pod = True,
        capytaine_pod_diameter = 2.0 * 0.0254,          # [m] Ø2 in (Fig. 20)
        capytaine_pod_length = 37.7 * 0.0254,           # [m] config G TOTAL: 12 in nose + 25.7 in body (Fig. 20)
        capytaine_pod_nose_ahead_of_le = 6.0 * 0.0254,  # [m] nose 6 in ahead of tip LE
        # (n_radial, n_theta, n_axial); (4,24,60) = 1.06% A-error vs (6,32,88) ref
        # ((3,16,40) was 2.92%) — same convergence study as mesh_n_span above.
        capytaine_pod_resolution = (4, 24, 60),
        # Pod top ↔ strut tip clearance. A true 0-gap weld is NOT cleanly BEM-
        # solvable: exact contact → coincident panels → Green-function NaN;
        # overlap+clip → open hole → collapsed added mass. A minimal gap keeps two
        # clean closed bodies; its value barely changes A(ω) (5 mm ≈ 25 mm ≈ contact).
        capytaine_pod_gap = 0.005,
        capytaine_pod_weld = False,
        # Coarser BEM ω grid: the strut+pod mesh (~2000 panels) costs ~7.5 s per
        # ω to build the Delhommeau matrix. A(ω) is smooth, so ~54 points (dense
        # in the 5–80 rad/s wet/flutter band) interpolate it well and keep the
        # BEM recompute to ~7 min instead of ~26 min.
        omega_list = np.unique(np.concatenate([
            np.linspace(0.1, 6.0, 6),
            np.linspace(6.0, 80.0, 40),
            np.linspace(80.0, 120.0, 8),
        ])),
        # Strut root/clamp is above the water; only the span below the waterline
        # (wetted_distance_from_root along the swept EA) is wetted. wetted_distance_from_root
        # is an arc-length along the 15°-swept EA; the loft projects it onto the vertical span
        # by cos(sweep), so the free surface sits at z = -wetted_distance·cos(sweep) — matching
        # the beam/aerogrid root height (z = -0.245 m, not -0.254 m).
        wetted_distance_from_root = wetted_distance_from_root,
        free_surface_elevation = -wetted_distance_from_root * np.cos(np.deg2rad(sweep_angle_deg)),
        water_depth = np.inf,
        depth = [0.0], #np.linspace(0.01, 0.5, 30),
        depth_index = 0,

        # ── Plotting ──────────────────────────────────────────────────────
        plot_aerobeam=True,
        plot_log_decrement_vg=True,
        show_stiffness_damping_contributions_plot=True,
        plot_DLM_participants=True,
        vf_hertz=True,
        v_knots=True,
        plot_added_mass_capytaine_vs_strip=True,
        plot_stiffness_damping_contributions=True,
        dimensionless_vgvf_results=True,
        plot_mode_shapes=True,
        classify_modes=True,
        plot_normalized_mode_shapes=True,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,

        caero_x1=-xea_factor*ort_chord+wetted_distance_from_root*np.sin(np.deg2rad(sweep_angle_deg)),
        caero_y1=wetted_distance_from_root*np.cos(np.deg2rad(sweep_angle_deg)),
        caero_z1=0,

        caero_x4=-xea_factor*ort_chord+ort_beam_length*np.sin(np.deg2rad(sweep_angle_deg)),
        caero_y4=ort_beam_length*np.cos(np.deg2rad(sweep_angle_deg)),
        caero_z4=0,

        # ── Section coordinates blunt strut SECT. A-A (chord normalized) ──
        # c = 12 in, t = 0.4 in (max thickness) => t/c = 1/30; symmetric about z/c = 0
        # 0 <= x/c <= 0.5: circular arcs (LE at z/c = 0); 0.5 < x/c <= 1: flat faces
        # Loop: LE -> pressure (lower) -> blunt TE cap -> suction (upper) -> LE
        raw=np.array([
            [0.0000,     -0.00000000],
            [0.0125,     -0.00082379],
            [0.0250,     -0.00162663],
            [0.0500,     -0.00316952],
            [0.0750,     -0.00462871],
            [0.1000,     -0.00600427],
            [0.1500,     -0.00850463],
            [0.2000,     -0.01067093],
            [0.2500,     -0.01250347],
            [0.3000,     -0.01400249],
            [0.3500,     -0.01516818],
            [0.4000,     -0.01600071],
            [0.4500,     -0.01650018],
            [0.5000,     -0.01666667],
            [0.6000,     -0.01666667],
            [0.7000,     -0.01666667],
            [0.8000,     -0.01666667],
            [0.9000,     -0.01666667],
            [0.9500,     -0.01666667],
            [1.0000,     -0.01666667],
            [1.0000,     -0.00555556],
            [1.0000,     0.00555556],
            [1.0000,     0.01666667],
            [0.9500,     0.01666667],
            [0.9000,     0.01666667],
            [0.8000,     0.01666667],
            [0.7000,     0.01666667],
            [0.6000,     0.01666667],
            [0.5000,     0.01666667],
            [0.4500,     0.01650018],
            [0.4000,     0.01600071],
            [0.3500,     0.01516818],
            [0.3000,     0.01400249],
            [0.2500,     0.01250347],
            [0.2000,     0.01067093],
            [0.1500,     0.00850463],
            [0.1000,     0.00600427],
            [0.0750,     0.00462871],
            [0.0500,     0.00316952],
            [0.0250,     0.00162663],
            [0.0125,     0.00082379],
            [0.0000,     0.00000000],
        ]),
    )