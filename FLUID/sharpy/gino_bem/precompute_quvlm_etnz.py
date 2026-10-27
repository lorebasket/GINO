"""GAF UVLM del braccio ETNZ AC75, proiettate sui modi di GINO (via nodale).

Stessa procedura di `precompute_quvlm_ac37.py`. Frame di SHARPy = frame di GINO. I nodi del boot
interpolano i nodi del braccio di GINO lungo z; l'ala di GINO e' rigida (RBE2 + travi 1e12), quindi i
nodi dell'ala e del collegamento seguono la punta del braccio con cinematica rigida (controllo: scarto
rispetto ai nodi dell'ala di GINO stampato a video).
  dpsi = T(psi0)^-1 C_BA(psi0)^T th_A

    [TNZ_LE=..] sharpy python sharpy/gino_bem/precompute_quvlm_etnz.py --gino <dry_modes.npz> --nm 10 --out <npz>
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
from gino_bem.etnz_tfoil import ETNZTFoil, GRIDS, WAKE_MATCH, RHO_W, LE

KN = 0.514444
# Le rotazioni di GINO devono essere il vettore di rotazione destrorso (beam_rotations_physical=True nella
# config di tnz_multibody). Con la convenzione storica sono -theta e la massa consistente e' sbagliata in un
# piano di flessione (27/10/2026): il controllo dell'ala rigida in gino_to_sharpy ferma il calcolo.
RIGID_TOL = 1e-5


def build(grid, nm, tag):
    N, Nw = GRIDS[grid]
    route = Path(__file__).resolve().parent / "cases"
    route.mkdir(exist_ok=True)
    ws = ETNZTFoil(N=N, N_wing=Nw, u_inf=1.0, route=str(route) + "/", case_name=tag)
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    dt = WAKE_MATCH
    mstar = int(round(float(os.environ.get("ETNZ_WAKE_LEN", "1.0")) / dt))
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
    print("[etnz] LE %+.3f m, dt = passo di scia %.1f mm, mstar %d (scia %.2f m), superfici M = %s"
          % (LE, 1e3 * dt, mstar, mstar * dt, list(ws.surface_m)))
    data = sharpy.sharpy_main.main(["", str(route / (tag + ".sharpy"))])
    return ws, data


def gino_to_sharpy(ws, data, g):
    pos, sub, Phi = g["pos"], g["sub"].astype(str), g["Phi"]
    nm = Phi.shape[1]
    U = Phi.reshape(len(pos), 6, nm)
    arm = np.flatnonzero(sub == "arm_spline")
    tipG = int(arm[np.argmin(pos[arm, 2])])
    low = arm[pos[arm, 2] < 0.6]
    low = low[np.argsort(pos[low, 2])]

    def rigid(n_G, rG):
        """Cinematica rigida dal nodo n_G di GINO al punto rG."""
        d = rG - pos[n_G]
        u = U[n_G].copy()
        u[:3] += np.cross(U[n_G, 3:].T, d).T
        return u

    # controllo: l'ala di GINO segue davvero la punta del braccio come corpo rigido
    fo = np.flatnonzero(np.char.startswith(sub, "foil"))
    err = max(np.abs(rigid(tipG, pos[i]) - U[i]).max() for i in fo) / np.abs(U[fo]).max()
    print("[etnz] ala di GINO vs cinematica rigida dalla punta del braccio: scarto relativo %.1e" % err)
    if err > RIGID_TOL:
        raise SystemExit("[etnz] ERRORE: l'ala non segue la punta del braccio con rotazioni destrorse (scarto %.1e): "
                         "modi con la convenzione storica? serve beam_rotations_physical=True" % err)

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
        rG = np.array([ws.x[n], ws.y[n], ws.z[n]])
        if n < ws.n_boot_node and n != ws.tip_idx:
            u = np.array([[np.interp(rG[2], pos[low, 2], U[low, c, m]) for m in range(nm)] for c in range(6)])
        else:
            u = rigid(tipG, rG)
        psi0 = psi_node[n]
        dpsi = np.linalg.solve(algebra.crv2tan(psi0), algebra.crv2rotation(psi0).T @ u[3:])
        out[6 * j:6 * j + 3] = u[:3]
        out[6 * j + 3:6 * j + 6] = dpsi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gino", type=Path, required=True)
    ap.add_argument("--grid", default="w20")
    ap.add_argument("--nm", type=int, default=10)
    ap.add_argument("--v-ref-kn", type=float, default=40.0)
    ap.add_argument("--k-max", type=float, default=60.0)
    ap.add_argument("--n-k", type=int, default=81)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    V = a.v_ref_kn * KN
    k = k_grid(a.k_max, a.n_k)
    g = dict(np.load(a.gino, allow_pickle=True))
    ws, dn = build(a.grid, a.nm, "etnz_le%+.3f_nm%d_%s" % (LE, a.nm, a.grid))
    Phi_s = gino_to_sharpy(ws, dn, g)[:, :a.nm]
    Q, info = gaf(dn, V, k, Phi=Phi_s)
    i_lo = int(np.argmin(np.abs(k - 0.5)))
    print("[etnz] Q(k=%.2f) diag %s" % (k[i_lo], np.round(np.diag(Q[i_lo]), 4)))
    i_hi = int(np.argmin(np.abs(k - 30.0)))
    print("[etnz] A_uvlm diag (k=%.0f) %s" % (k[i_hi], np.round(-0.5 * RHO_W * np.diag(Q[i_hi]).real / k[i_hi] ** 2, 4)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, k_list=k, Q=Q, V_ref_m_s=V, rho=RHO_W, qinf_ref=0.5 * RHO_W * V ** 2,
             F_ref=info["F_ref"], grid=a.grid, case="etnz_le%+.3f" % LE, nm=a.nm, source_gino_npz=str(a.gino),
             dt_scaled=info["dt"], n_states=info["states"], f_gino=g["f"][:a.nm],
             wake_match_mm=1e3 * WAKE_MATCH, surface_m=np.asarray(ws.surface_m),
             convention=("F_modal = qinf * Q(k_pa) * q_gino ; k_pa = omega/V [1/m]; "
                         "base = modi del braccio ETNZ di GINO (massa unitaria); proiezione nodale"))
    print("[etnz] salvato", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
