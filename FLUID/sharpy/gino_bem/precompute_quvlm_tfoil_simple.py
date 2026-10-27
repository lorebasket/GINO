"""GAF UVLM del T-foil semplificato, proiettate sui modi di GINO (via nodale).

Stessa procedura di `precompute_quvlm_tfoil.py` (AC38): UVLM di SHARPy in coordinate
nodali, ingressi/uscite proiettati con le forme di GINO espresse nei gdl di SHARPy.
Qui i nodi di SHARPy e di GINO stanno negli STESSI punti (asse elastico, nessun
offset al giunto), quindi la conversione e' solo di frame:
  x_A = x_G, y_A = -z_G, z_A = y_G         (u_A = R u_G, th_A = R th_G)
  dpsi = T(psi0)^-1 C_BA(psi0)^T th_A

    conda run -n sharpy python sharpy/gino_bem/precompute_quvlm_tfoil_simple.py --gino <dry_modes.npz> --nm 8 --out <npz>
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
from gino_bem.precompute_quvlm_tfoil import gaf, k_grid
from gino_bem.tfoil_simple import TfoilSimple, GRIDS, G

KN = 0.514444
R_GA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def build(grid, nm, tag):
    M, N, Ms, Ne = GRIDS[grid]
    route = Path(__file__).resolve().parent / "cases"
    route.mkdir(exist_ok=True)
    ws = TfoilSimple(M=M, N=N, Mstar_fact=Ms, u_inf=1.0, N_elev=Ne, route=str(route) + "/", case_name=tag)
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    dt = ws.main_chord / M
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader", "StaticUvlm", "Modal", "LinearAssembler"]
    ws.config["StaticUvlm"] = ws.config["StaticCoupled"]["aero_solver_settings"].dict()
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": int(Ms * M),
        "freestream_dir": ["1", "0", "0"], "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {"u_inf": 1.0, "u_inf_direction": ["1", "0", "0"], "dt": dt}}
    ws.config["Modal"] = {"NumLambda": max(30, 2 * nm), "rigid_body_modes": False, "print_matrices": False,
                          "write_modes_vtk": False, "use_undamped_modes": True, "save_data": False}
    bs = {"modal_projection": "off", "inout_coords": "nodes", "discrete_time": "on", "newmark_damp": 0.5e-4,
          "discr_method": "newmark", "dt": dt, "proj_modes": "undamped", "use_euler": "off", "num_modes": nm,
          "print_info": "off", "gravity": "off"}
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": bs,
            "aero_settings": {"dt": dt, "ScalingDict": {"length": 0.5 * ws.c_ref, "speed": 1.0, "density": ws.rho},
                              "integr_order": 2, "density": ws.rho, "remove_predictor": "on", "use_sparse": "on",
                              "remove_inputs": ["u_gust"], "added_mass_removal": "none"},
            "uvlm_filename": "", "track_body": "on", "use_euler": "off"}}
    ws.config.write()
    data = sharpy.sharpy_main.main(["", str(route / (tag + ".sharpy"))])
    return ws, data


def gino_to_sharpy(ws, data, g):
    pos, sub, Phi = g["pos"], g["sub"].astype(str), g["Phi"]
    nm = Phi.shape[1]
    U = Phi.reshape(len(pos), 6, nm)
    rud = sub == "rudder"
    tipG = int(np.flatnonzero(rud)[np.argmax(pos[rud, 1])])
    # radice dell'elevatore: con l'elevatore innestato sul timone e' la punta del timone; con
    # l'elevatore spostato (TFS_EDX) e' il nodo comune alle due semiali (in GINO etichettato col
    # primo ramo che lo contiene)
    el = np.flatnonzero(np.char.startswith(sub, "elevator") | (sub == "link"))
    rootG = int(el[np.argmin(np.abs(pos[el, 2]) + 10 * np.abs(pos[el, 1] - pos[tipG, 1]) +
                             1e3 * (np.abs(pos[el, 0] - pos[tipG, 0]) < 1e-9))]) if el.size else tipG
    if np.abs(pos[rootG, 0] - pos[tipG, 0]) < 1e-9:
        rootG = tipG
    def branch(lab):
        m = np.flatnonzero(sub == lab)
        idx = np.unique(np.concatenate([[rootG], m]))           # la radice e' comune alle due semiali
        key = np.abs(pos[idx, 2]); o = np.argsort(key)
        return key[o], U[idx[o]]
    br = {lab: branch(lab) for lab in ("elevator_stbd", "elevator_port")}
    o = np.argsort(pos[rud, 1]); yr, Ur = pos[rud, 1][o], U[rud][o]
    node_lab = {}
    for lab, gl in ws.elev_nodes.items():
        for k, n in enumerate(gl[1:]):
            node_lab[int(n)] = (lab, ws.elev_s[k + 1])
    vdof = np.asarray(data.structure.vdof)
    out = np.zeros((6 * int((vdof >= 0).sum()), nm))
    ts = data.structure.timestep_info[0]
    psi_node = {}
    for ie in range(data.structure.num_elem):
        for il in range(data.structure.num_node_elem):
            psi_node.setdefault(int(data.structure.connectivities[ie, il]), ts.psi[ie, il])
    for n in range(ws.num_node_tot):
        j = int(vdof[n])
        if j < 0:
            continue
        if n in node_lab:
            lab, s = node_lab[n]; kk, UU = br[lab]
        elif getattr(ws, "link_elem", None) is not None and n in ws.conn_glob[ws.link_elem][1:]:
            # nodi del collegamento rigido: radice dell'elevatore (estremo) o punto medio, cinematica
            # rigida dalla radice
            lab = "elevator_stbd"; s = 0.0; kk, UU = br[lab]
        else:
            s = ws.z[n]; kk, UU = yr, Ur
        u = np.array([[np.interp(s, kk, UU[:, c, m]) for m in range(nm)] for c in range(6)])
        uA = R_GA @ u[:3]; thA = R_GA @ u[3:]
        psi0 = psi_node[n]
        dpsi = np.linalg.solve(algebra.crv2tan(psi0), algebra.crv2rotation(psi0).T @ thA)
        out[6 * j:6 * j + 3] = uA
        out[6 * j + 3:6 * j + 6] = dpsi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gino", type=Path, required=True)
    ap.add_argument("--grid", default="n40c12e16")
    ap.add_argument("--nm", type=int, default=8)
    ap.add_argument("--v-ref-kn", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=40.0)
    ap.add_argument("--n-k", type=int, default=81)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    V = a.v_ref_kn * KN
    k = k_grid(a.k_max, a.n_k)
    g = dict(np.load(a.gino, allow_pickle=True))
    # pelo libero e modi nel nome: run paralleli non si sovrascrivono i file del caso
    ws, dn = build(a.grid, a.nm, "tfoil_simple_" + G.tag().replace(".", "p") + "_wl%d_nm%d" % (round(1e3 * G.WL), a.nm))
    Phi_s = gino_to_sharpy(ws, dn, g)[:, :a.nm]
    Q, info = gaf(dn, V, k, Phi=Phi_s)
    i_lo = int(np.argmin(np.abs(k - 0.5)))
    print("[tfoil_simple] Q(k=%.2f) diag %s" % (k[i_lo], np.round(np.diag(Q[i_lo]), 4)))
    # massa apparente UVLM sul piatto in k: A = 0.5 rho Re(Q) / k^2 (diagonale)
    i_hi = int(np.argmin(np.abs(k - 30.0)))
    print("[tfoil_simple] A_uvlm diag (k=%.0f) %s" % (k[i_hi], np.round(-0.5 * G.RHO_W * np.diag(Q[i_hi]).real / k[i_hi] ** 2, 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, k_list=k, Q=Q, V_ref_m_s=V, rho=G.RHO_W, qinf_ref=0.5 * G.RHO_W * V ** 2,
             F_ref=info["F_ref"], grid=a.grid, case="tfoil_simple", nm=a.nm, source_gino_npz=str(a.gino),
             dt_scaled=info["dt"], n_states=info["states"], f_gino=g["f"][:a.nm],
             convention=("F_modal = qinf * Q(k_pa) * q_gino ; k_pa = omega/V [1/m]; "
                         "base = modi del T di GINO (massa unitaria); proiezione nodale"))
    print("[tfoil_simple] salvato", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
