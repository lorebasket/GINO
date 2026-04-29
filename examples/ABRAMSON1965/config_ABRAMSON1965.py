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

    V_low     = np.linspace(2,  10,  8)
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

        # ── Rayleigh damping ──────────────────────────────────────────────
        rayleigh_target_zetas=(-0.01, -0.01),

        # ── Post-processing ───────────────────────────────────────────────
        save_matrices=True,
        plot_mode_shapes=True,
        classify_modes=True,
        selective_dofs=None,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,
    )
