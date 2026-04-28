import numpy as np
import os


def get_config(AnalysisConfig, FSI_path, QJJ_path):
    sonata_name      = 'GOLAND'
    attack_angle_deg = 0.05
    fluid            = 'air'
    aerogrid_name    = 'GOLAND_v2'

    _base = AnalysisConfig('GOLAND_sonata')
    nspan, nchord = _base.nspan, _base.nchord

    qjj_dir = os.path.join(
        FSI_path,
        f"{aerogrid_name}_{fluid}_alpha{attack_angle_deg}_nspan{nspan}_nchord{nchord}_klist_new"
    )
    aerogrid_path = os.path.join(qjj_dir, "aerogrid.npz")

    return AnalysisConfig(
        'GOLAND_sonata',
        sonata_name=sonata_name,
        chord=1.8228,
        fluid=fluid,
        alpha_deg=attack_angle_deg,
        alpha_r=np.deg2rad(attack_angle_deg),
        V_list=np.linspace(100, 180, 30),
        modes_to_analyze=[0, 1, 2],
        plot_aerobeam=True,
        nspan=nspan,
        nchord=nchord,
        qjj_dir=qjj_dir,
        aerogrid_path=aerogrid_path,
    )
