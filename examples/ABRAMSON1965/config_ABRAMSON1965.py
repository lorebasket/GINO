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

    V_low     = np.linspace(5,  10,  5)
    V_flutter = np.linspace(10, 18, 30)
    V_high    = np.linspace(18, 20,  5)

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
        pitch=0,

        # ── Material & mass ───────────────────────────────────────────────
        rho_s=7000,
        mu=72.10,
        radius_gyration=0.512,
        v=0.3,

        # ── EA / CG locations ─────────────────────────────────────────────
        xcm_factor=0.512,
        xea_factor=0.25,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=7845,
        GJ=1718,
        k_i22=0.3,
        k_i33=0.7,
        EA=1.0e8,
        EIzz=1.0e9,
        GAx=1.0e8,
        GAz=1.0e8,

        # ── Rayleigh damping ──────────────────────────────────────────────
        rayleigh_target_zetas=(-0.01, -0.01),

        # ── Analysis struct params ────────────────────────────────────────
        modal_dir = os.path.join(FSI_path, "output_data", "ABRAMSON1965"),
        prefix = "ABRAMSON1965_dry_egv",

        # ── Flutter sweep ─────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.concatenate([V_low, V_flutter, V_high]),
        num_modes_flutter_egv=2,

        # ── Roger RFA ─────────────────────────────────────────────────────
        added_mass_strip_theory=True,
        roger_fit=True,
        roger_fit_modes=3,
        k_list=np.linspace(0.001, 50, 400),
        n_lag=1,
        blag=np.linspace(4, 10, 1),

        # ── Capytaine sweep ──────────────────────────────────────────────
        mesh_path = os.path.join(FSI_path, "FLUID", "capytaine", "ABRAMSON1965", "ABRAMSON1965_mesh.vtu"),
        mesh_n_span = 30,
        mesh_n_chord = 30,
        omega_list = np.linspace(1.0, 120, 60),
        free_surface_elevation = 0.0,
        water_depth = 1.0,
        depth = np.linspace(0.01, 0.5, 30),
        
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