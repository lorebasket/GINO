import os
"""GAF UVLM del timone AC38 a T, proiettate DIRETTAMENTE sui modi del T di GINO.

PERCHE' NON LA CONGRUENZA DI `precompute_quvlm.py`. Quella accoppia ogni modo di
GINO a un modo di SHARPy lungo un'unica coordinata di span (spostamento e
rotazione della trave). Nel T timone ed elevatore si muovono insieme e non c'e'
una coordinata di span unica. Qui si costruisce l'UVLM di SHARPy in coordinate
NODALI (`modal_projection: off`) e lo si proietta con le forme di GINO espresse
nei gradi di liberta' di SHARPy, esattamente come SHARPy fa con le sue
(`linearaeroelastic.py`: in = blkdiag(Phi, Phi), out = Phi^T). La struttura
dentro SHARPy non entra nel risultato.

CONVERSIONE GINO -> SHARPy, per nodo libero:
  spostamento  u_A = R u_G
  rotazione    dpsi = T(psi0)^-1 C_BA(psi0) R th_G     (dth_B = T dpsi, cfr. Kdisp)
con R la rotazione propria fra frame di GINO (prima del diedro) e frame A:
x_A = x_G, y_A = -z_G, z_A = y_G (lato +y_A di SHARPy = semiala di GINO con z_G < 0).

CONTROLLO (--check): con Phi = modi propri di SHARPy la via nodale deve dare le
stesse GAF della proiezione modale di SHARPy.

  python gino_bem/precompute_quvlm_tfoil.py --gino <npz modi GINO> [--check]
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sharpy.sharpy_main
import sharpy.utils.algebra as algebra
import sharpy.linear.src.libss as libss
from gino_bem.ac38_tfoil import AC38TFoil, GRIDS

KN = 0.514444
RHO = 997.0
# x_G = -(x_CAD - 777) e x_A sono ENTRAMBI nel verso della corrente (bordo d'attacco a
# x_CAD massimo); rotazione propria: x_A = x_G, y_A = -z_G, z_A = y_G
R_GA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # u_A = R u_G
if os.environ.get('AC38_BUG_RGA'):   # diagnostica: segno vecchio (errato)
    R_GA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def build(grid, modal, nm, tag):
    M, N, Ms, Ne, *fd = GRIDS[grid]
    fdt = float(fd[0]) if fd else 1.0           # passo di scia = pannello / fdt
    route = Path(__file__).resolve().parent / "cases"
    route.mkdir(exist_ok=True)
    ws = AC38TFoil(M=M, N=N, Mstar_fact=Ms, u_inf=1.0, N_elev=Ne,
                   route=str(route) + "/", case_name=tag)
    # stato stazionario di linearizzazione (LV 22/10/2026): nel modello SHARPy del T, alpha ruota il corpo
    # attorno all'asse della lama (= angolo di barra, carica la lama) e yaw attorno all'asse laterale
    # (= incidenza dell'elevatore, carica le semiali); verificato sulle forze di StaticUvlm
    ws.alpha = float(os.environ.get("AC38_BARRA_DEG", "0") or 0.0)
    ws.yaw = float(os.environ.get("AC38_ELEV_DEG", "0") or 0.0)
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    dt = ws.main_chord / M / fdt
    Lw = float(os.environ.get("AC38_WAKE_MATCH", "0") or 0.0) * 1e-3
    if Lw > 0:                                   # scia adattata: passo = lunghezza dei pannelli in corda
        dt = Lw
        fdt = ws.main_chord / M / Lw
        Ms = float(os.environ.get("AC38_WAKE_LEN", "1.0")) / ws.main_chord     # mstar = Ms M fdt = lunghezza / Lw
    # niente StaticCoupled: a incidenza zero su superfici piane il riferimento e'
    # la geometria indeformata, e con la struttura di comodo dell'elevatore lo
    # statico non lineare non converge sulla griglia fine (residuo 1e8)
    # AC38_WAKE_BODY=1: scia piana allineata col corpo (asse x del corpo, cioe' lungo le corde) invece che con la
    # corrente. Con angolo di barra la scia lungo la corrente si stacca dalla mezzeria dell'elevatore e ne attraversa
    # i pannelli: effetto a "interruttore", non proporzionale al carico (GAF non lineare nell'angolo, 22/10/2026).
    # La scia congelata lungo il corpo e' l'ipotesi classica per piccoli angoli (come il VLM armonico).
    wake_dir = ["1", "0", "0"]
    if os.environ.get("AC38_WAKE_BODY"):
        xb = algebra.quat2rotation(ws.quat) @ np.array([1.0, 0.0, 0.0])
        wake_dir = [repr(float(v)) for v in xb / np.linalg.norm(xb)]
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader", "StaticUvlm", "Modal", "LinearAssembler"]
    ws.config["StaticUvlm"] = ws.config["StaticCoupled"]["aero_solver_settings"].dict()
    # AC38_VORTEX_RADIUS [mm]: raggio di nucleo dei vortici (default SHARPy 1e-6 m). Serve quando una scia passa
    # vicino ai punti di controllo di un'altra superficie (barra: scia della lama sul piano dell'elevatore)
    rv = float(os.environ.get("AC38_VORTEX_RADIUS", "0") or 0.0) * 1e-3
    if rv > 0:
        ws.config["StaticUvlm"].update({"vortex_radius": rv, "vortex_radius_wake_ind": rv})
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": int(round(Ms * M * fdt)),
        "freestream_dir": ["1", "0", "0"], "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {"u_inf": 1.0, "u_inf_direction": wake_dir, "dt": dt}}
    ws.config["Modal"] = {"NumLambda": max(30, 2 * nm), "rigid_body_modes": False,
                          "print_matrices": False, "write_modes_vtk": False,
                          "use_undamped_modes": True, "save_data": False}
    bs = {"modal_projection": "on" if modal else "off",
          "inout_coords": "modes" if modal else "nodes",
          "discrete_time": "on", "newmark_damp": 0.5e-4, "discr_method": "newmark",
          "dt": dt, "proj_modes": "undamped", "use_euler": "off", "num_modes": nm,
          "print_info": "off", "gravity": "off"}
    ws.config["LinearAssembler"] = {
        "linear_system": "LinearAeroelastic",
        "linear_system_settings": {
            "beam_settings": bs,
            "aero_settings": {"dt": dt, "ScalingDict": {"length": 0.5 * ws.c_ref, "speed": 1.0,
                                                        "density": ws.rho},
                              "integr_order": 2, "density": ws.rho, "remove_predictor": "on",
                              "use_sparse": "on", "remove_inputs": ["u_gust"],
                              "added_mass_removal": "none",
                              **({"vortex_radius": rv} if rv > 0 else {})},
            "uvlm_filename": "", "track_body": "on", "use_euler": "off"}}
    ws.config.write()
    data = sharpy.sharpy_main.main(["", str(route / (tag + ".sharpy"))])
    return ws, data


def k_grid(k_max, n_k):
    if n_k == 60:
        # griglia leggera (LV): 20 punti in 0.02-2, 20 in 2.2-8 (zona del flutter, k ~ 4-6), 19 fino a k_max
        return np.unique(np.concatenate([[0.0], np.linspace(0.02, 2.0, 20),
                                         np.linspace(2.2, 8.0, 20),
                                         np.linspace(8.5, k_max, 19)]))
    return np.unique(np.concatenate([[0.0], np.linspace(0.02, 2.0, 34),
                                     np.linspace(2.2, 8.0, 22),
                                     np.linspace(8.5, k_max, max(n_k - 57, 2))]))


def gaf(data, V_ref, k_list, Phi=None, Phi_out=None):
    """Q(k) per ampiezza modale. Phi=None: l'UVLM e' gia' proiettato da SHARPy.
    Phi_out: proiezione in uscita se diversa da quella in ingresso (immagine del fondo scafo)."""
    ls = data.linear.linear_system
    ls.update(float(V_ref))
    ss = ls.uvlm.ss
    facts = ss.ScalingFacts if hasattr(ss, 'ScalingFacts') else ls.uvlm.sys.ScalingFacts
    F_ref = float(facts["dyn_pressure"]) * float(facts["length"]) ** 2
    t_ref = float(facts["length"]) / float(V_ref)
    dimfac = F_ref / float(facts["dyn_pressure"])
    if Phi is not None:
        nd, nm = Phi.shape
        Bin = np.zeros((ss.inputs, 2 * nm)); Bin[:nd, :nm] = Phi; Bin[nd:2 * nd, nm:] = Phi
        Cout = np.zeros((nm, ss.outputs)); Cout[:, :nd] = (Phi if Phi_out is None else Phi_out).T
        B = ss.B @ Bin; C = Cout @ ss.C; D = Cout @ ss.D @ Bin
        red = libss.StateSpace(ss.A, B, C, D, dt=ss.dt)
    else:
        red = ss; nm = ss.outputs
    w_s = k_list * V_ref * t_ref
    H = red.freqresp(w_s)
    Q = np.array([np.conj(dimfac * (H[:nm, :nm, i] + 1j * w * H[:nm, nm:2 * nm, i]))
                  for i, w in enumerate(w_s)])
    return Q, dict(F_ref=F_ref, t_ref=t_ref, dt=float(ss.dt), states=ss.states)


def gino_to_sharpy(ws, data, g):
    """Modi di GINO -> vettore dei gdl liberi di SHARPy (6 per nodo: pos_A, psi)."""
    pos, sub, Phi = g["pos"], g["sub"], g["Phi"]
    nm = Phi.shape[1]
    U = Phi.reshape(len(pos), 6, nm)
    rud = sub == "rudder"
    o = np.argsort(pos[rud, 1])
    yr, Ur, Pr = pos[rud, 1][o], U[rud][o], pos[rud][o]
    br = {}
    for lab in ("elevator_port", "elevator_stbd"):
        m = sub == lab
        oo = np.argsort(np.abs(pos[m, 2]))
        br[lab] = (np.abs(pos[m, 2])[oo], U[m][oo], pos[m][oo])
    tip_g = Ur[-1]
    vdof = np.asarray(data.structure.vdof)
    nd = 6 * int((vdof >= 0).sum())
    out = np.zeros((nd, nm))
    ts = data.structure.timestep_info[0]
    psi_node = {}
    for ie in range(data.structure.num_elem):
        for il in range(data.structure.num_node_elem):
            psi_node.setdefault(int(data.structure.connectivities[ie, il]), ts.psi[ie, il])
    s0 = ws.z[0]
    # distanza nodo SHARPy - punto GINO (frame GINO), salvata nel file GAF per il controllo di posizione
    offsets = gino_to_sharpy.offsets = []
    node_lab = {}
    uth = {}
    for lab, gl in (("elevator_stbd", ws.elev_nodes["port"]), ("elevator_port", ws.elev_nodes["stbd"])):   # +y_A <-> z_G < 0
        for k, n in enumerate(gl[1:]):
            node_lab[int(n)] = (lab, ws._ye[k + 1] * 1e-3)
    for n in range(ws.num_node_tot):
        j = int(vdof[n])
        if j < 0:
            continue
        if n in node_lab:
            lab, ye = node_lab[n]
            yy, UU, PP = br[lab]
            u = np.array([[np.interp(ye, yy, UU[:, c, m]) for m in range(nm)] for c in range(6)])
            rg = np.array([np.interp(ye, yy, PP[:, c]) for c in range(3)])
        else:
            s = ws.z[n] - s0                       # span dalla cima, come y_G di GINO
            u = np.array([[np.interp(s, yr, Ur[:, c, m]) for m in range(nm)] for c in range(6)])
            rg = np.array([np.interp(s, yr, Pr[:, c]) for c in range(3)])
        # il nodo di SHARPy non sta dove sta il punto di GINO (radice dell'elevatore
        # innestata sulla punta della lama, offset di 41 mm trascurato): si trasporta
        # lo spostamento con la cinematica rigida della sezione, u + th x (r_S - r_G)
        rs = R_GA.T @ np.array([ws.x[n], ws.y[n], ws.z[n] - s0])
        offsets.append(np.r_[n, rs - rg])
        u[:3] += np.cross(u[3:].T, (rs - rg)[None, :]).T
        uA = R_GA @ u[:3]; thA = R_GA @ u[3:]
        uth[n] = (uA, thA)
        psi0 = psi_node[n]
        Cab = algebra.crv2rotation(psi0); Tan = algebra.crv2tan(psi0)
        dpsi = np.linalg.solve(Tan, Cab.T @ thA)
        out[6 * j:6 * j + 3] = uA
        out[6 * j + 3:6 * j + 6] = dpsi
    # immagine del fondo scafo (AC38_IMAGE): riflessione z_A -> 2 z_f - z_A. Spostamento
    # S u, rotazione (pseudovettore) -S th; per la superficie libera (fs) tutto col segno -.
    gino_to_sharpy.out_mask = np.ones(nd, bool)
    if getattr(ws, "image_map", None):
        S = np.diag([1.0, 1.0, -1.0]); sg = ws.image_sign
        for n_img in list(ws.image_map) + [ws.image_link_mid]:
            j = int(vdof[n_img])
            if j >= 0:
                gino_to_sharpy.out_mask[6 * j:6 * j + 6] = False       # forze dell'immagine: non sono carichi veri
        for n_img, n_r in ws.image_map.items():
            j = int(vdof[n_img])
            if j < 0:
                continue
            uA_r, thA_r = uth.get(n_r, (np.zeros((3, nm)), np.zeros((3, nm))))   # nodo incastrato: fermo
            uA = sg * (S @ uA_r); thA = -sg * (S @ thA_r)
            psi0 = psi_node[n_img]
            Cab = algebra.crv2rotation(psi0); Tan = algebra.crv2tan(psi0)
            out[6 * j:6 * j + 3] = uA
            out[6 * j + 3:6 * j + 6] = np.linalg.solve(Tan, Cab.T @ thA)
        # il nodo di mezzo del collegamento non ha pannelli: niente moto
        j = int(vdof[ws.image_link_mid])
        if j >= 0:
            out[6 * j:6 * j + 6] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gino", type=Path, required=True)
    ap.add_argument("--grid", default="n48c14e20")
    ap.add_argument("--nm", type=int, default=6)
    ap.add_argument("--v-ref-kn", type=float, default=60.0)
    ap.add_argument("--k-max", type=float, default=40.0)
    ap.add_argument("--n-k", type=int, default=81)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    V = a.v_ref_kn * KN
    k = k_grid(a.k_max, a.n_k)
    g = dict(np.load(a.gino, allow_pickle=True))
    if a.check:
        nmc = 6
        _, dm = build(a.grid, True, nmc, "tfoil_chk_modal")
        Qm, _ = gaf(dm, V, k[:6])
        ws, dn = build(a.grid, False, nmc, "tfoil_chk_nodal")
        U = np.asarray(dn.linear.linear_system.beam.sys.U)[:, :nmc] \
            if hasattr(dn.linear.linear_system.beam.sys, 'U') and dn.linear.linear_system.beam.sys.U is not None \
            else np.real(np.asarray(dn.structure.timestep_info[-1].modal['eigenvectors']))[:, :nmc]
        Qn, _ = gaf(dn, V, k[:6], Phi=U)
        err = np.abs(Qn - Qm).max() / np.abs(Qm).max()
        print("[check] scarto relativo max via nodale vs proiezione SHARPy: %.2e" % err)
        print("[check] Q modale  k=%.2f diag %s" % (k[3], np.round(np.diag(Qm[3]), 4)))
        print("[check] Q nodale  k=%.2f diag %s" % (k[3], np.round(np.diag(Qn[3]), 4)))
        return 0
    tag = "tfoil_nodal" + (("_dx%03d" % round(float(os.environ["AC38_ELEV_DX"]))) if os.environ.get("AC38_ELEV_DX") else "") \
        + (("_img" + os.environ["AC38_IMAGE"]) if os.environ.get("AC38_IMAGE") else "") \
        + (("_ext" + os.environ["AC38_TIP_EXT"]) if os.environ.get("AC38_TIP_EXT") else "") \
        + ("_sh" + Path(os.environ["AC38_ELEV_SHIFT"]).stem if os.environ.get("AC38_ELEV_SHIFT") else "") \
        + ("_rs" + Path(os.environ["AC38_RUD_SHIFT"]).stem if os.environ.get("AC38_RUD_SHIFT") else "")   # nome unico: casi in parallelo
    ws, dn = build(a.grid, False, a.nm, tag)
    Phi_s = gino_to_sharpy(ws, dn, g)[:, :a.nm]
    Phi_o = Phi_s * gino_to_sharpy.out_mask[:, None]
    Q, info = gaf(dn, V, k, Phi=Phi_s, Phi_out=Phi_o)
    i_lo = int(np.argmin(np.abs(k - 0.5)))
    print("[tfoil] Q(k=%.2f) =\n%s" % (k[i_lo], np.array2string(Q[i_lo], precision=4, suppress_small=True)))
    out = a.out or (Path(__file__).resolve().parents[3] / "FLUID" / "PanelAero" / "Qjj" /
                    f"Quvlm_AC38_TFOIL_water_{a.grid}_nm{a.nm}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    # griglia UVLM effettivamente usata (frame CAD) e scarti SHARPy-GINO: li legge
    # examples/AC38_TFOIL/check_uvlm_position.py
    ts0 = dn.structure.timestep_info[0]
    C_GA = algebra.quat2rotation(ts0.quat)
    zeta_cad = {}
    for i, z in enumerate(dn.aero.timestep_info[0].zeta):
        p = np.einsum("ji,jab->iab", C_GA, z - ts0.for_pos[:3, None, None]).reshape(3, -1).T
        p = np.c_[0.777 - p[:, 0], p[:, 1], 0.565 - (p[:, 2] - ws.z[0])]
        zeta_cad["zeta_cad_%d" % i] = p.T.reshape(z.shape)
    np.savez(out, k_list=k, Q=Q, V_ref_m_s=V, rho=RHO, qinf_ref=0.5 * RHO * V ** 2,
             elev_dx_mm=float(os.environ.get("AC38_ELEV_DX", "0") or 0.0),
             # AC38_ELEV_SHIFT: traslazione per sezione (check_uvlm_position.py la riapplica)
             elev_shift=(np.loadtxt(os.environ["AC38_ELEV_SHIFT"], delimiter=",", comments="#", ndmin=2)
                         if os.environ.get("AC38_ELEV_SHIFT") else np.zeros((0, 2))),
             # AC38_RUD_SHIFT: traslazione per sezione del timone (z_mm, dx_mm)
             rud_shift=(np.loadtxt(os.environ["AC38_RUD_SHIFT"], delimiter=",", comments="#", ndmin=2)
                        if os.environ.get("AC38_RUD_SHIFT") else np.zeros((0, 2))),
             image=os.environ.get("AC38_IMAGE", ""),
             barra_deg=float(os.environ.get("AC38_BARRA_DEG", "0") or 0.0),
             elev_deg=float(os.environ.get("AC38_ELEV_DEG", "0") or 0.0),
             wake_match_mm=float(os.environ.get("AC38_WAKE_MATCH", "0") or 0.0),
             vortex_radius_mm=float(os.environ.get("AC38_VORTEX_RADIUS", "0") or 0.0),
             wake_body=bool(os.environ.get("AC38_WAKE_BODY")),
             tip_ext_mm=float(os.environ.get("AC38_TIP_EXT", "0") or 0.0),
             surface_m=np.asarray(ws.surface_m), surface_group=np.asarray(ws.surface_group),
             gino_offsets=np.array(gino_to_sharpy.offsets), **zeta_cad,
             F_ref=info["F_ref"], grid=a.grid, case="ac38_tfoil", nm=a.nm,
             source_gino_npz=str(a.gino), dt_scaled=info["dt"], n_states=info["states"],
             f_gino=g["f"][:a.nm],
             convention=("F_modal = qinf * Q(k_pa) * q_gino ; k_pa = omega/V [1/m]; "
                         "base = modi del T di GINO (massa unitaria); proiezione nodale"))
    print("[tfoil] salvato", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
