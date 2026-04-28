import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    fluid            = 'water'
    attack_angle_deg = 0.0

    _base = AnalysisConfig('1x1grid')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(
        FSI_path,
        f"1x1grid_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_klist_new"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    return AnalysisConfig(
        name='1x1grid',
        blade_name='1x1grid',
        fluid=fluid,

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=1.0,
        chord=1.0,
        chord_root=1.0,
        chord_tip=1.0,

        # ── Material ──────────────────────────────────────────────────────
        rho_s=2700,
        mu=35.709721,
        v=0.31,
        i11=8.64,
        i22=0.1 * 8.64,
        i33=0.9 * 8.64,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIyy=9.77221e6,
        GJ=9.87581e5,
        EIzz=9.77221e6,
        EA=1.0e12,
        GAy=1.0e12,
        GAz=1.0e12,
        xcm_factor=0.43,
        xea_factor=0.33,

        # ── Flutter sweep ─────────────────────────────────────────────────
        save_matrices=True,
        pitch=0,
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        plot_aerobeam=True,

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        aerogrid_path=aerogrid_path,
    )
