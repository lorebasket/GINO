#!/usr/bin/env python3
"""Banco di prova del nostro UVLM lineare (SHARPy, stessa catena di precompute_quvlm_tfoil.py) su un'ala
rettangolare rigida in traslazione e beccheggio, per misurare l'errore di discretizzazione in funzione di k.

Ala: corda c = 1, allungamento 8 (come la lama AC38), asse di beccheggio al 30 % della corda, profilo piano,
U = 1. Uscite: forze generalizzate totali Q_hh, Q_ha, Q_ah, Q_aa (F = qinf Q q, convenzione DT: Im < 0 = smorzante)
contro k_b = omega b / U (b = c/2). Varianti:
  M<m>      m pannelli in corda, passo di scia = pannello (dt = c/m), scia 10 corde
  M14w6     come AC38 alla radice: 14 pannelli, scia 6 corde di riferimento
  M14tip    come AC38 alla PUNTA: 14 pannelli legati ma passo di scia 3.33 volte il pannello (dt = c/4.2),
            84 pannelli di scia (= stessa lunghezza assoluta della scia AC38 rispetto alla corda di punta)
  M14w3     14 pannelli, scia di 3 corde
  M14fine   14 pannelli legati, passo di scia 3.3 volte PIU' FINE del pannello, scia 3 corde

    python gino_bem/bench_theodorsen_uvlm.py VARIANTE [VARIANTE ...] --out DIR
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SHARPY_ROOT))
import sharpy.sharpy_main                                    # noqa: E402
import sharpy.utils.algebra as algebra                       # noqa: E402
import sharpy.cases.templates.flying_wings as fw             # noqa: E402

from gino_bem.precompute_quvlm_tfoil import gaf              # noqa: E402

AR, EA, NSPAN = 8.0, 0.30, 32


def variante(nome):
    """-> (M, mstar, dt) con c = U = 1."""
    if nome == "M14w6":
        return 14, 6 * 14, 1.0 / 14
    if nome == "M14tip":
        return 14, 6 * 14, 1.0 / 4.2
    if nome == "M14w3":                   # scia corta, 3 corde
        return 14, 3 * 14, 1.0 / 14
    if nome == "M14fine":                 # come la RADICE AC38 con passo adatto alla punta: scia 3.3x piu' fine
        return 14, int(round(3 * 14 * 3.3)), 1.0 / (14 * 3.3)
    m = int(nome[1:])
    return m, 10 * m, 1.0 / m


def build(nome, route):
    M, mstar, dt = variante(nome)
    ws = fw.FlyingWing(M=M, N=NSPAN, Mstar_fact=10, u_inf=1.0, alpha=0.0, rho=1.0, b_ref=AR, main_chord=1.0,
                       aspect_ratio=AR, n_surfaces=2, route=str(route) + "/", case_name="bench_" + nome)
    ws.main_ea = EA
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader", "StaticUvlm", "Modal", "LinearAssembler"]
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["StaticUvlm"] = ws.config["StaticCoupled"]["aero_solver_settings"].dict()
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": int(mstar),
        "freestream_dir": ["1", "0", "0"], "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {"u_inf": 1.0, "u_inf_direction": ["1", "0", "0"], "dt": dt}}
    ws.config["Modal"] = {"NumLambda": 20, "rigid_body_modes": False, "print_matrices": False,
                          "write_modes_vtk": False, "use_undamped_modes": True, "save_data": False}
    bs = {"modal_projection": "off", "inout_coords": "nodes", "discrete_time": "on", "newmark_damp": 0.5e-4,
          "discr_method": "newmark", "dt": dt, "proj_modes": "undamped", "use_euler": "off", "num_modes": 10,
          "print_info": "off", "gravity": "off"}
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": bs,
            "aero_settings": {"dt": dt, "ScalingDict": {"length": 0.5, "speed": 1.0, "density": 1.0},
                              "integr_order": 2, "density": 1.0, "remove_predictor": "on", "use_sparse": "on",
                              "remove_inputs": ["u_gust"], "added_mass_removal": "none"},
            "uvlm_filename": "", "track_body": "on", "use_euler": "off"}}
    ws.config.write()
    data = sharpy.sharpy_main.main(["", str(route / ("bench_" + nome + ".sharpy"))])
    return ws, data


def phi_rigido(data):
    """Due 'modi' nodali: traslazione verticale unitaria (z_A) e beccheggio unitario a cabrare attorno a y_A."""
    vdof = np.asarray(data.structure.vdof)
    nd = 6 * int((vdof >= 0).sum())
    ts = data.structure.timestep_info[0]
    psi = {}
    for ie in range(data.structure.num_elem):
        for il in range(data.structure.num_node_elem):
            psi.setdefault(int(data.structure.connectivities[ie, il]), ts.psi[ie, il])
    P = np.zeros((nd, 2))
    for n in range(len(vdof)):
        j = int(vdof[n])
        if j < 0:
            continue
        Cab = algebra.crv2rotation(psi[n]); Tan = algebra.crv2tan(psi[n])
        P[6 * j:6 * j + 3, 0] = [0.0, 0.0, 1.0]
        # rotazione +y_A: il bordo d'attacco (x < x_asse) sale, cioe' cabra
        P[6 * j + 3:6 * j + 6, 1] = np.linalg.solve(Tan, Cab.T @ np.array([0.0, 1.0, 0.0]))
    return P


def theodorsen(kb, a=2 * EA - 1.0):
    """GAF 2D per unita' di apertura, convenzione F = qinf Q q con qinf = 1/2, b = 1/2, h su (z), alpha a cabrare.
    Ritorna (Q_hh, Q_ha, Q_ah, Q_aa) con forza generalizzata = portanza (su) e momento a cabrare attorno all'asse."""
    from scipy.special import hankel2
    b = 0.5
    k = np.asarray(kb, float); kk = np.where(k == 0, 1e-9, k)
    C = hankel2(1, kk) / (hankel2(1, kk) + 1j * hankel2(0, kk))
    C = np.where(k == 0, 1.0, C)
    s = 1j * 2 * k                     # U = 1, b = 1/2 -> omega = k/b
    rho = 1.0
    # forma standard (Fung), h positivo in basso, L in alto, M a cabrare attorno all'asse (x = a b dal centro):
    #   L = pi rho b^2 (h_dd + U al_d - b a al_dd) + 2 pi rho U b C (h_d + U al + b (1/2 - a) al_d)
    #   M = pi rho b^2 (b a h_dd - U b (1/2 - a) al_d - b^2 (1/8 + a^2) al_dd)
    #       + 2 pi rho U b^2 (a + 1/2) C (h_d + U al + b (1/2 - a) al_d)
    # per z = -h i coefficienti della traslazione cambiano segno.
    Lh = np.pi * rho * b**2 * s**2 + 2 * np.pi * rho * b * C * s
    La = np.pi * rho * b**2 * (s - b * a * s**2) + 2 * np.pi * rho * b * C * (1 + b * (0.5 - a) * s)
    Mh = np.pi * rho * b**2 * (b * a * s**2) + 2 * np.pi * rho * b**2 * (a + 0.5) * C * s
    Ma = (np.pi * rho * b**2 * (-b * (0.5 - a) * s - b**2 * (1 / 8 + a**2) * s**2)
          + 2 * np.pi * rho * b**2 * (a + 0.5) * C * (1 + b * (0.5 - a) * s))
    L_z, M_z = -Lh, -Mh
    q = 0.5
    # forza generalizzata della traslazione z = portanza (su); del beccheggio = momento a cabrare
    return np.array([[L_z, La], [M_z, Ma]]).transpose(2, 0, 1) / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("varianti", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    route = _SHARPY_ROOT / "gino_bem" / "cases"; route.mkdir(exist_ok=True)
    kb = np.concatenate([[0.0], np.linspace(0.02, 1.6, 40)])
    for v in a.varianti:
        ws, data = build(v, route)
        P = phi_rigido(data)
        Q, info = gaf(data, 1.0, 2 * kb, Phi=P)        # k_pa = omega/U = 2 k_b
        Q = np.conj(Q)                                  # convenzione del p-k (aero_imag_flip): Im < 0 smorzante
        np.savez(a.out / ("bench_%s.npz" % v), kb=kb, Q=Q, states=info["states"], dt=info["dt"], AR=AR, EA=EA)
        print("%s: %d stati; Q_aa(k_b=0.5) = %s" % (v, info["states"], Q[np.argmin(abs(kb - 0.5)), 1, 1]), flush=True)
    np.savez(a.out / "theodorsen_2d.npz", kb=kb, Q=theodorsen(kb) * AR)


if __name__ == "__main__":
    main()
