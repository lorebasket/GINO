import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    sonata_name      = 'hollowell-1982'
    sonata_case_name = 'm2_prova'
    attack_angle_deg = 0.0
    fluid            = 'air'
    chord            = 0.076
    aerogrid_name    = 'hollowell'

    _base = AnalysisConfig('hollowell')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(
        FSI_path,
        f"{aerogrid_name}_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_klist_new"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    V_low     = np.linspace(1,  26,  30)
    V_flutter = np.linspace(26, 34, 160)
    V_high    = np.linspace(34, 50,  20)

    return AnalysisConfig(
        'hollowell',
        sonata_name=sonata_name,
        sonata_case_name=sonata_case_name,
        chord=chord,
        fluid=fluid,
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.concatenate([V_low, V_flutter, V_high]),
        modes_to_analyze=[0, 1, 2],
        plot_aerobeam=True,
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        aerogrid_path=aerogrid_path,
    )
