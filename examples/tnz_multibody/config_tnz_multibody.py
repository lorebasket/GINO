import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    aerogrid_name    = 'MULTI_tnz_boot_tnz_foil_dx_tnz_foil_sx'
    fluid            = 'water'
    attack_angle_deg = 0.0
    cg_wing_offsett  =  0.1   # >0 moves aft, <0 moves forward from arm SC
    le_wing_offset   = -0.1   # >0 moves aft, <0 moves forward from arm SC
    added_mass_foil_mu = 0.0

    _base = AnalysisConfig('tnz_multibody')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(
        QJJ_path,
        f"{aerogrid_name}_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic_leoffset{le_wing_offset:.1f}"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    return AnalysisConfig(
        'tnz_multibody',
        
        # ── Geometry ──────────────────────────────────────────────────────
        chord=0.4,
        boot_length=0.300,
        foil_span=2.000,

        # ── Fluid ─────────────────────────────────────────────────────────
        fluid=fluid,
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        pitch=0,
        cor=[3.46944695e-18, -2.63945203e+00, 3.15636665e+00],

        # ── Material ──────────────────────────────────────────────────────
        rho_s=795,

        # ── Arm beam properties ───────────────────────────────────────────
        mu_arm_constant=False,
        mu_arm=37.2,
        EIxx_scale=1,
        EIyy_scale=1,
        GK_scale=1,
        GAx_scale=1,
        GAz_scale=1,
        nu_EG=2.5,
        kappa=0.85,

        # ── Foil-wing sectional properties ────────────────────────────────
        foil_mass_model='hybrid',
        mu=0.001 + added_mass_foil_mu,
        i11=0.5,
        i22=2.0,
        i33=2.3,
        EIxx=1e12,
        GJ=1e12,
        EIzz=1e12,
        EA=1e12,
        GAx=1e12,
        GAz=1e12,
        xcm_factor=0.5,
        xea_factor=0.5,

        # ── Concentrated point mass ───────────────────────────────────────
        foil_mass=849,
        foil_mass_location=[0, 0.0, -525],

        # ── Foil geometry ─────────────────────────────────────────────────
        _spine_rot_deg=32.66640714,
        dihedral_angle=20.0,
        foil_chordwise_offset=cg_wing_offsett,
        cg_wing_mass_offset_flag=True,

        # ── Element count ─────────────────────────────────────────────────
        n_nodes_foil=10,

        # ── RBE2 rigid connector ──────────────────────────────────────────
        use_rbe2_connector=True,
        rbe2_stiffness_scale=1e12,
        rbe2_connector_type='rigid_links',
        rbe2_offset_tolerance=1e-9,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=qjj_dir,
        aerogrid_path=aerogrid_path,

        # ── Flutter sweep ─────────────────────────────────────────────────
        V_list=np.linspace(5, 45, 40),
        num_modes_flutter_egv=4,
    )
