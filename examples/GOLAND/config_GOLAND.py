import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    attack_angle_deg = 2
    fluid = 'air'

    _base = AnalysisConfig('GOLAND')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir  = os.path.join(QJJ_path, f"GOLAND_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    vjj_dir  = os.path.join(QJJ_path, f"vjj_GOLAND_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    V_low     = np.linspace(100, 160, 30)
    V_flutter = np.linspace(160, 170, 10)
    V_high    = np.linspace(170, 180, 10)

    return AnalysisConfig(
        name='GOLAND',
        blade_name='GOLAND',
        fluid=fluid,

        # ── Geometry ──────────────────────────────────────────────────────
        beam_length=6.096,
        chord=1.8288,
        chord_root=1.8288,
        chord_tip=1.8288,
        pitch=0,
        chordwise_strip_y=3.048,    # 50% span

        # ── Material & mass ───────────────────────────────────────────────
        rho_s=2700,
        mu=35.71,
        v=0.31,
        i11=8.64,
        i22=0.1 * 8.64,
        i33=0.9 * 8.64,

        # ── Stiffness ─────────────────────────────────────────────────────
        EIxx=9.77221e6,
        GJ=0.987581e6,
        EIzz=1e2 * 9.77221e6,
        EA=1.0e9,
        GAx=1.0e9,
        GAz=1.0e9,

        # ── EA / CG locations ─────────────────────────────────────────────
        xcm_factor=0.43,
        xea_factor=0.33,

        # ── Flutter sweep ─────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.concatenate([V_low, V_flutter, V_high]),
        num_modes_flutter_egv=2,

        # ── Rayleigh damping ──────────────────────────────────────────────
        rayleigh_target_zetas=(0.0, -0.005),

        # ── Aerodynamic grid ──────────────────────────────────────────────
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        vjj_dir=vjj_dir,
        aerogrid_path=aerogrid_path,
    )
