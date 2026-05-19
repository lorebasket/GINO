import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    fluid            = 'water'
    attack_angle_deg = 0.0

    _base = AnalysisConfig('wing01')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(QJJ_path, f"wing01_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_quartic")
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    return AnalysisConfig(
        'wing01',
        sonata_name='wing01',
        sonata_case_name='wing01',
        fluid=fluid,

        # ── Geometry ──────────────────────────────────────────────────────
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        pitch=0,

        # ── Methodology ───────────────────────────────────────────────────
        pk_method=True,
        roger_fit=False,
        RFA_PK_method=False,

        # ── Flutter sweep ─────────────────────────────────────────────────
        V_list=np.linspace(20, 50, 20),
        num_modes_flutter_egv=2,
        plot_aerobeam=True,
        aerogrid_path=aerogrid_path,
        qjj_dir=qjj_dir,
        vjj_dir=qjj_dir,
    )
