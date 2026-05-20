import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    fluid            = 'water'
    attack_angle_deg = 0.0

    _base = AnalysisConfig('ABRAMSON1965')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(QJJ_path, f"ABRAMSON1965_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    vjj_dir = os.path.join(QJJ_path, f"vjj_ABRAMSON1965_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    V_low     = np.linspace(2.5,  16,  50)
    V_flutter = np.linspace(16, 20, 50)
    V_high    = np.linspace(20, 23,  10)

    return AnalysisConfig(
        name='ABRAMSON1965',
        blade_name='ABRAMSON1965',
        fluid=fluid,

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=0.762,
        chord=0.3048,
        chord_root=0.3048,
        chord_tip=0.3048,
        thickness_factor=0.12,
        chordwise_strip_y=0.381,
        # Post-build rotation about +X (deg); see post_pitch_utils.
        # pitch_rotate_beam / pitch_rotate_aerogrid: only trave, only DLM grid, or both (default).
        pitch=-90,
        pitch_rotate_beam=True,
        pitch_rotate_aerogrid=False,
        beam_span_axis=2, # 2 Z axis if pitch = -90 deg

        # ── Material & mass ───────────────────────────────────────────────
        rho_s=7000,
        mu=72.14, # total weight 121.2lb = 54.97 kg → 72.14 kg/m
        radius_gyration=0.512,
        v=0.3,

        # ── EA / CG locations ─────────────────────────────────────────────
        xcm_factor=0.512, # 0.25 * chord + x_alpha * (chord / 2) = 0.25 * chord + 0.524 * (chord / 2) = 0.512 * chord
        xea_factor=0.25,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=9.76e3, # 3.40×106 lbf\cdotpin2→9.76×103 N\cdotpm2
        GJ= 2.79e3, # 0.973×106 lbf\cdotpin2→2.79×103 N\cdotpm2
        k_i22=0.3,
        k_i33=0.7,
        EA=1.0e8,
        EIzz=1.0e9,
        GAx=1.0e8,
        GAz=1.0e8,

        # ── Rayleigh damping ──────────────────────────────────────────────
        rayleigh_target_zetas=(0.09, 0.09), # high structural damping due to geometrical experiment manufacturing

        # ── Im(Q) scale (RFA fit unscaled; applied in C_eff and optionally lag forcing) ──
        aero_im_Q_scale=1.0,  # try 0.3–0.7 if aero destabilizes too early vs Abramson ~48 kn
        aero_im_Q_scale_lags=False,

        # ── Empirical fluid damping (Abramson Fig. 5: ~Δδ≈4 s⁻¹ above measured-Q curve) ──
        empirical_fluid_damping=True,
        empirical_fluid_model='abramson_delta',
        empirical_fluid_delta_add_s=5.0, # It is recommended to use a value of 5.0 s⁻¹ for the empirical fluid damping.


        # ── Analysis struct params ────────────────────────────────────────
        modal_dir = os.path.join(FSI_path, "output_data", "ABRAMSON1965"),
        prefix = "ABRAMSON1965_dry_egv",

        # ── Flutter sweep ─────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.concatenate([V_low, V_flutter, V_high]),

        # ── Roger RFA lag poles ───────────────────────────────────────────
        # PanelAero remains in k = omega/V [1/m].  These are dimensionless
        # beta = k * semichord, converted in the solver to blag [1/m].
        n_lag=4,
        blag=np.array([0.5, 1.0, 1.5, 2.0]),
        #rfa_blag_dimensionless=False,
        #rfa_blag_ref_length='semichord',

        # ── Capytaine sweep ──────────────────────────────────────────────
        capytaine_results_dir = os.path.join(FSI_path, "FLUID", "capytaine", "ABRAMSON1965", "results_modal_radiation"),
        mesh_path = os.path.join(FSI_path, "FLUID", "capytaine", "ABRAMSON1965", "ABRAMSON1965_mesh.vtu"),
        mesh_n_span = 30,
        mesh_n_chord = 30,
        omega_list = np.linspace(0.1, 150, 300),
        free_surface_elevation = 0.0,
        water_depth = np.inf,
        depth = [0.0508, 0.052], #np.linspace(0.01, 0.5, 30),
        depth_index = 0,

        # ── Plotting ──────────────────────────────────────────────────────
        plot_log_decrement_vg=True,
        show_stiffness_damping_contributions_plot=True,
        plot_DLM_participants=True,
        vf_hertz=True,
        v_knots=True,
        plot_added_mass_capytaine_vs_strip=True,
        plot_stiffness_damping_contributions=True,
        dimensionless_vgvf_results=True,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,

        # ── Section coordinates (normalized by chord length) ──────────────
        raw=np.array([
            [1.0000,     0.00120],
            [0.9500,     0.01415],
            [0.9000,     0.02517],
            [0.8000,     0.04199],
            [0.7000,     0.05269],
            [0.6000,     0.05835],
            [0.5000,     0.06000],
            [0.4000,     0.05855],
            [0.3000,     0.05417],
            [0.2000,     0.04664],
            [0.1500,     0.04135],
            [0.1000,     0.03457],
            [0.0750,     0.03032],
            [0.0500,     0.02509],
            [0.0250,     0.01805],
            [0.0125,     0.01292],
            [0.0000,     0.00000],
            [0.0125,     -0.01292],
            [0.0250,     -0.01805],
            [0.0500,     -0.02509],
            [0.0750,     -0.03032],
            [0.1000,     -0.03457],
            [0.1500,     -0.04135],
            [0.2000,     -0.04664],
            [0.3000,     -0.05417],
            [0.4000,     -0.05855],
            [0.5000,     -0.06000],
            [0.6000,     -0.05835],
            [0.7000,     -0.05269],
            [0.8000,     -0.04199],
            [0.9000,     -0.02517],
            [0.9500,     -0.01415],
            [1.0000,     -0.00120]
        ]),
    )