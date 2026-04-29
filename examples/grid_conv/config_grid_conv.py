import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    fluid            = 'water'
    attack_angle_deg = 0.0

    _base = AnalysisConfig('grid_conv')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(
        FSI_path,
        f"grid_conv_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_klist_new"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    return AnalysisConfig(
        name='grid_conv',
        blade_name='grid_conv',
        fluid=fluid,

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=3.0,
        chord=0.5,
        chord_root=0.5,
        chord_tip=0.5,
        dihedral_angle=0.0,
        le_points={'root': [0.0, 0.0, 0.0],   'tip': [0.0, 3000.0, 0.0]},
        te_points={'root': [+500, 0.0, 0.0],   'tip': [+500, 3000.0, 0.0]},
        cg_points={'root': [+250, 0.0, 0.0],   'tip': [+250, 3000.0, 0.0]},
        thickness=50,   # [mm]
        cor=[250, 0.0, 0.0],

        # ── Material ──────────────────────────────────────────────────────
        rho_s=2700,
        mu=35.709721,
        v=0.31,
        i11=8.64,
        i22=0.1 * 8.64,
        i33=0.9 * 8.64,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=0.7e6,
        EIzz=0.7e6 * 1e2,
        GJ=0.7e5,
        EA=1.0e9,
        GAy=1.0e9,
        GAz=1.0e9,
        xcm_factor=0.5,
        xea_factor=0.4,

        # ── Flutter sweep ─────────────────────────────────────────────────
        save_matrices=True,
        pitch=0,
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.linspace(10, 50, 25),
        num_modes_flutter_egv=2,
        num_modes=8,

        # ── Hybrid non-circulatory operator ───────────────────────────────
        hybrid_nc_operator=True,
        capytaine_singlebody=True,

        # ── Post-processing ───────────────────────────────────────────────
        plot_aerobeam=True,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=qjj_dir,
        aerogrid_path=aerogrid_path,
    )
