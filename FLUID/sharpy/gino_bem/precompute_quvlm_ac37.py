"""GAF UVLM del timone AC37, proiettate sui modi di GINO (via nodale).

Stessa procedura di `precompute_quvlm_tfoil_simple.py`: UVLM di SHARPy in coordinate
nodali, ingressi/uscite proiettati con le forme di GINO espresse nei gdl di SHARPy.
I nodi di SHARPy stanno sulle stesse linee dei centri di taglio di GINO
(ac37_geom.rudder_axis / elev_nodes), quindi basta interpolare lungo l'apertura; i nodi
del timone sotto l'ultima stazione di GINO (Z = 2.75, la superficie arriva a 2.813) e il
nodo medio del collegamento seguono la punta del timone con cinematica rigida.
  x_A = x_G, y_A = -z_G, z_A = y_G         (u_A = R u_G, th_A = R th_G)
  dpsi = T(psi0)^-1 C_BA(psi0)^T th_A

    conda run -n sharpy python sharpy/gino_bem/precompute_quvlm_ac37.py --gino <dry_modes.npz> --nm 10 --out <npz>
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
from gino_bem.precompute_quvlm_tfoil import gaf, k_grid
from gino_bem.ac37_tfoil import AC37TFoil, GRIDS, G, WAKE_MATCH

KN = 0.514444
R_GA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def build(grid, nm, tag):
    M, N, Ms, Ne = GRIDS[grid]
    route = Path(__file__).resolve().parent / "cases"
    route.mkdir(exist_ok=True)
    ws = AC37TFoil(M=M, N=N, Mstar_fact=1, u_inf=1.0, N_elev=Ne, route=str(route) + "/", case_name=tag)
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    dt = ws.main_chord / M
    mstar = int(round(float(os.environ.get("AC37_WAKE_LEN", "0.8")) / dt)) if WAKE_MATCH > 0 else Ms
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader", "StaticUvlm", "Modal", "LinearAssembler"]
    ws.config["StaticUvlm"] = ws.config["StaticCoupled"]["aero_solver_settings"].dict()
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": mstar,
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
            "aero_settings": {"dt": dt, "ScalingDict": {"length": 0.5 * 0.2, "speed": 1.0, "density": ws.rho},
                              "integr_order": 2, "density": ws.rho, "remove_predictor": "on", "use_sparse": "on",
                              "remove_inputs": ["u_gust"], "added_mass_removal": "none"},
            "uvlm_filename": "", "track_body": "on", "use_euler": "off"}}
    ws.config.write()
    print("[ac37] dt = passo di scia %.1f mm, mstar %d (scia %.2f m), superfici M = %s"
          % (1e3 * dt, mstar, mstar * dt, list(ws.surface_m)))
    data = sharpy.sharpy_main.main(["", str(route / (tag + ".sharpy"))])
    return ws, data


def gino_to_sharpy(ws, data, g):
    pos, sub, Phi = g["pos"], g["sub"].astype(str), g["Phi"]
    nm = Phi.shape[1]
    U = Phi.reshape(len(pos), 6, nm)
    rud = np.flatnonzero(sub == "rudder")
    o = np.argsort(pos[rud, 1]); rud = rud[o]
    tipG = rud[-1]
    el = np.flatnonzero(np.char.startswith(sub, "elevator"))
    rootG = int(el[np.argmin(np.abs(pos[el, 2]))])

    def branch(lab):
        m = np.flatnonzero(sub == lab)
        idx = np.unique(np.concatenate([[rootG], m]))           # la radice e' comune alle due semiali
        key = np.abs(pos[idx, 2]); oo = np.argsort(key)
        return key[oo], U[idx[oo]]
    br = {lab: branch(lab) for lab in ("elevator_stbd", "elevator_port")}

    def rigid(n_G, rG):
        """Cinematica rigida dal nodo n_G di GINO al punto rG (frame di GINO)."""
        d = rG - pos[n_G]
        u = U[n_G].copy()
        u[:3] += np.cross(U[n_G, 3:].T, d).T
        return u

    node_lab = {}
    for lab, gl in ws.elev_nodes.items():
        for k, n in enumerate(gl[1:]):
            node_lab[int(n)] = (lab, ws.elev_Y[k + 1])
    link_nodes = set() if ws.link_elem is None else {int(i) for i in ws.conn_glob[ws.link_elem][1:]}
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
        rG = R_GA.T @ np.array([ws.x[n], ws.y[n], ws.z[n]])
        if n in node_lab:
            lab, s = node_lab[n]; kk, UU = br[lab]
            u = np.array([[np.interp(s, kk, UU[:, c, m]) for m in range(nm)] for c in range(6)])
        elif n == ws.elev_root and n != ws.tip_idx:
            u = U[rootG]
        elif n in link_nodes or rG[1] > pos[tipG, 1] + 1e-9:
            u = rigid(tipG, rG)
        else:
            u = np.array([[np.interp(rG[1], pos[rud, 1], U[rud, c, m]) for m in range(nm)] for c in range(6)])
        uA = R_GA @ u[:3]; thA = R_GA @ u[3:]
        psi0 = psi_node[n]
        dpsi = np.linalg.solve(algebra.crv2tan(psi0), algebra.crv2rotation(psi0).T @ thA)
        out[6 * j:6 * j + 3] = uA
        out[6 * j + 3:6 * j + 6] = dpsi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gino", type=Path, required=True)
    ap.add_argument("--grid", default="n56c16e18")
    ap.add_argument("--nm", type=int, default=10)
    ap.add_argument("--v-ref-kn", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=60.0)
    ap.add_argument("--n-k", type=int, default=81)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    V = a.v_ref_kn * KN
    k = k_grid(a.k_max, a.n_k)
    g = dict(np.load(a.gino, allow_pickle=True))
    ws, dn = build(a.grid, a.nm, "ac37_" + G.tag().replace(".", "p") + "_imm%g_nm%d_%s" % (G.IMM, a.nm, a.grid))
    Phi_s = gino_to_sharpy(ws, dn, g)[:, :a.nm]
    Q, info = gaf(dn, V, k, Phi=Phi_s)
    i_lo = int(np.argmin(np.abs(k - 0.5)))
    print("[ac37] Q(k=%.2f) diag %s" % (k[i_lo], np.round(np.diag(Q[i_lo]), 4)))
    i_hi = int(np.argmin(np.abs(k - 30.0)))
    print("[ac37] A_uvlm diag (k=%.0f) %s" % (k[i_hi], np.round(-0.5 * G.RHO_W * np.diag(Q[i_hi]).real / k[i_hi] ** 2, 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, k_list=k, Q=Q, V_ref_m_s=V, rho=G.RHO_W, qinf_ref=0.5 * G.RHO_W * V ** 2,
             F_ref=info["F_ref"], grid=a.grid, case="ac37_" + G.tag(), nm=a.nm, source_gino_npz=str(a.gino),
             dt_scaled=info["dt"], n_states=info["states"], f_gino=g["f"][:a.nm], imm=G.IMM, z_wl=G.Z_WL,
             wake_match_mm=1e3 * WAKE_MATCH, surface_m=np.asarray(ws.surface_m),
             convention=("F_modal = qinf * Q(k_pa) * q_gino ; k_pa = omega/V [1/m]; "
                         "base = modi del timone AC37 di GINO (massa unitaria); proiezione nodale"))
    print("[ac37] salvato", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
