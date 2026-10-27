"""Modulo di caso AC38 I_B_500 (RevD) per ``precompute_quvlm.py``.

Fornisce il contratto che il driver si aspetta: ``GRIDS``, ``build_case`` e
l'aggancio alla base modale. Costruisce il caso SHARPy del timone, esegue il
flusso fino a ``LinearAssembler`` e restituisce il ``data`` da cui si estrae il
ROM UVLM.

NOTA SUL MODELLO STRUTTURALE. Il modello dentro SHARPy e' incastrato alla
boccola inferiore, perche' SHARPy non sa esprimere i vincoli veri del timone
(boccola inferiore: ux, uy, uz, thz bloccati, mx e my liberi; superiore: solo
ux, uy). Misurato, quell'approssimazione costa +10.8 % sul primo modo e +50.1 %
sul secondo. Per questo NON e' il modello strutturale dell'analisi: serve solo a
generare la base su cui costruire il ROM aerodinamico. La struttura vera, con i
vincoli di plan_00, sta in GINO e viaggia per NPZ
(``modal_basis_dry.npz`` / NPZ di radiazione modale di Capytaine).
"""
from pathlib import Path

import numpy as np
import sharpy.sharpy_main

from gino_bem.ac38_rudder import AC38Rudder, GRIDS, default_ac38_capy_npz  # noqa: F401

KN = 0.514444
_SHARPY_ROOT = Path(__file__).resolve().parents[1]
_FSI_ROOT = _SHARPY_ROOT.parents[1]


def _ac38_velocity_list_m_s(v_max_m_s: float, n: int = 40) -> np.ndarray:
    return np.linspace(1.0, float(v_max_m_s), int(n))


def build_case(nm: int, *, grid, added_mass_removal: str, rayleigh: bool,
               zetas, v_max_kn: float, case_tag: str):
    M, N, Ms = grid
    route = _SHARPY_ROOT / "gino_bem"
    (route / "cases").mkdir(parents=True, exist_ok=True)
    (route / "output").mkdir(exist_ok=True)

    ws = AC38Rudder(M=M, N=N, Mstar_fact=Ms, u_inf=1.0, alpha=0.0,
                    route=str(route / "cases") + "/", case_name=case_tag,
                    variant='IB500')
    ws.clean_test_files()
    ws.update_derived_params()
    ws.set_default_config_dict()
    ws.generate_aero_file()
    ws.generate_fem_file()

    vel = _ac38_velocity_list_m_s(v_max_kn * KN)
    u_inf = 1.0
    dt = ws.main_chord / M / u_inf

    ws.config["SHARPy"]["flow"] = [
        "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
    ]
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": int(Ms * M),
        "freestream_dir": ["1", "0", "0"],
        "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {
            "u_inf": u_inf, "u_inf_direction": ["1", "0", "0"], "dt": dt},
    }
    ws.config["Modal"] = {
        "NumLambda": max(30, 4 * nm), "rigid_body_modes": False,
        "print_matrices": False, "write_modes_vtk": False,
        "use_undamped_modes": True, "save_data": False,
    }
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": {"modal_projection": "on", "inout_coords": "modes",
                              "discrete_time": "on", "newmark_damp": 0.5e-4,
                              "discr_method": "newmark", "dt": dt,
                              "proj_modes": "undamped", "use_euler": "off",
                              "num_modes": nm, "print_info": "off",
                              "gravity": "off"},
            "aero_settings": {
                "dt": dt,
                # Il ROM si costruisce sul sistema UVLM ADIMENSIONALIZZATO:
                # senza ScalingDict, update_matrices_time_scale fallisce perche'
                # le matrici di riferimento scalate non esistono.
                "ScalingDict": {"length": 0.5 * ws.c_ref,
                                "speed": u_inf,
                                "density": ws.rho},
                "integr_order": 2,
                "density": ws.rho,
                "remove_predictor": "on",
                "use_sparse": "on",
                "remove_inputs": ["u_gust"],
                "added_mass_removal": added_mass_removal},
            "uvlm_filename": "", "track_body": "on", "use_euler": "off",
        },
    }
    ws.config.write()
    print(">>> AC38: griglia M,N,Ms = %s   added_mass_removal = %r   nm = %d"
          % (grid, added_mass_removal, nm), flush=True)
    data = sharpy.sharpy_main.main(["", str(route / "cases" / (case_tag + ".sharpy"))])
    return data, vel


def lifting_span_mask(data) -> np.ndarray:
    """Maschera sui nodi liberi: True dove ci sono pannelli UVLM.

    In I_B_500 la superficie portante e' TUTTA quella bagnata, da z = -90 mm al
    tip: la tabella Geometria_idro di RevD la definisce li'.
    """
    from gino_bem.ac38_rudder import Z_BEARING_UP
    pos = np.asarray(data.linear.tsstruct0.pos, dtype=float)
    bcs = np.asarray(data.structure.boundary_conditions).ravel()
    free = bcs != 1
    span_axis = int(np.argmax(pos.max(axis=0) - pos.min(axis=0)))
    s = pos[free, span_axis]
    s_foil = (Z_BEARING_UP - (-90.0)) * 1e-3
    return np.abs(s - s.min()) >= (s_foil - (Z_BEARING_UP - 565.0) * 1e-3) - 1e-6
