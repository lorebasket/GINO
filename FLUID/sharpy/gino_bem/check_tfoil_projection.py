"""Verifica del trasferimento dei modi GINO -> SHARPy per il T AC38.

  conda run -n sharpy python sharpy/gino_bem/check_tfoil_projection.py [--grid n48c14e20]

Tre controlli, nessuno dei quali usa R_GA come dato:

1. ROTAZIONE DALLE POSIZIONI. Le posizioni dei nodi di GINO e di SHARPy sono
   ancorate al CAD in modo indipendente (tavola sovrapposta). La rotazione che
   porta i nodi di GINO su quelli di SHARPy si ricava ai minimi quadrati
   (Kabsch) e si confronta con R_GA attuale e con quella vecchia. Gli
   spostamenti DEVONO ruotare con la stessa R delle posizioni.

2. SEGNO FISICO (teoria delle strisce quasi-statica). Per k -> 0 la forza
   generalizzata vale  Q_ij = -sum_strisce 2 pi c ds (theta_j x t).n (n.u_i(AC)),
   con t = direzione della corrente e n = normale della striscia (il segno di n
   si elide). Si calcola con i modi di GINO e la geometria dei pannelli, senza
   passare per la proiezione su SHARPy, e si confronta il SEGNO di Re Q(k=0)
   con le due tabelle UVLM (vecchia e nuova).

3. NORMALI DEI PANNELLI: per superficie, nel frame CAD.

Figure: per i primi modi, il modo di GINO e lo stesso modo trasferito su SHARPy
(R attuale e R vecchia) in tre viste ortogonali nel frame CAD, con la griglia
dei pannelli deformata; e la mappa delle normali.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sharpy.sharpy_main
from gino_bem.ac38_tfoil import AC38TFoil, GRIDS

import os
FSI = Path(__file__).resolve().parents[3]
RS = "" if os.environ.get("AC38_REV", "reva") == "legacy" else "_reva"   # revisione del modello (vedi config)
R_NEW = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
R_OLD = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
NM = 6


def build_geom(grid):
    M, N, Ms, Ne = GRIDS[grid]
    route = Path(__file__).resolve().parent / "cases"
    route.mkdir(exist_ok=True)
    ws = AC38TFoil(M=M, N=N, Mstar_fact=Ms, u_inf=1.0, N_elev=Ne,
                   route=str(route) + "/", case_name="tfoil_geomchk")
    ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
    ws.generate_aero_file(); ws.generate_fem_file()
    ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader"]
    ws.config["SHARPy"]["write_screen"] = "off"
    ws.config["AerogridLoader"] = {
        "unsteady": "off", "aligned_grid": "on", "mstar": 2,
        "freestream_dir": ["1", "0", "0"], "wake_shape_generator": "StraightWake",
        "wake_shape_generator_input": {"u_inf": 1.0, "u_inf_direction": ["1", "0", "0"],
                                       "dt": ws.main_chord / M}}
    ws.config.write()
    data = sharpy.sharpy_main.main(["", str(route / "tfoil_geomchk.sharpy")])
    return ws, data


# --- mappe nel frame CAD [m] (dall'HANDOFF, verificate sulla tavola) ---------
def gino_to_cad(p):                    # posizioni; per i vettori: senza traslazione
    return np.c_[0.777 - p[:, 0], -p[:, 2], 0.565 - p[:, 1]]


def gvec_to_cad(v):
    return np.c_[-v[:, 0], -v[:, 2], -v[:, 1]]


def a_to_cad(p, s0):
    return np.c_[0.777 - p[:, 0], p[:, 1], 0.565 - (p[:, 2] - s0)]


def avec_to_cad(v):
    return np.c_[-v[:, 0], v[:, 1], -v[:, 2]]


def pairs(ws, g):
    """Coppie (nodo SHARPy, punto GINO) con la stessa logica di gino_to_sharpy."""
    pos, sub = g["pos"], g["sub"]
    s0 = ws.z[0]
    out = []                                   # (n_sharpy, lab, coord di span)
    node_lab = {}
    for lab, gl in (("elevator_stbd", ws.elev_nodes["port"]), ("elevator_port", ws.elev_nodes["stbd"])):
        for k, n in enumerate(gl[1:]):
            node_lab[int(n)] = (lab, ws._ye[k + 1] * 1e-3)
    rud = sub == "rudder"
    for n in range(ws.num_node_tot):
        if n in node_lab:
            lab, ye = node_lab[n]
            m = sub == lab
            yy = np.abs(pos[m, 2]); o = np.argsort(yy)
            rg = np.array([np.interp(ye, yy[o], pos[m][o][:, c]) for c in range(3)])
            out.append((n, lab, rg))
        else:
            s = ws.z[n] - s0
            yr = pos[rud, 1]; o = np.argsort(yr)
            rg = np.array([np.interp(s, yr[o], pos[rud][o][:, c]) for c in range(3)])
            out.append((n, "rudder", rg))
    return out


def kabsch(P, Qp):
    """R, t con  Qp ~ R P + t  (rotazione propria)."""
    pc, qc = P.mean(0), Qp.mean(0)
    H = (P - pc).T @ (Qp - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def interp_mode(g, lab, ycoord, nm):
    pos, sub, Phi = g["pos"], g["sub"], g["Phi"]
    U = Phi.reshape(len(pos), 6, -1)[:, :, :nm]
    m = sub == lab
    yy = pos[m, 1] if lab == "rudder" else np.abs(pos[m, 2])
    o = np.argsort(yy)
    return np.array([[np.interp(ycoord, yy[o], U[m][o][:, c, j]) for j in range(nm)]
                     for c in range(6)]), \
        np.array([np.interp(ycoord, yy[o], pos[m][o][:, c]) for c in range(3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="n48c14e20")
    a = ap.parse_args()
    out = FSI / "output_data" / "AC38_TFOIL" / ("06_check_projection" + RS)
    out.mkdir(parents=True, exist_ok=True)
    g = dict(np.load(FSI / ("output_data/AC38_TFOIL/00_dry%s/tfoil_dry_modes.npz" % RS), allow_pickle=True))
    ws, data = build_geom(a.grid)
    s0 = ws.z[0]
    rep = []
    P = lambda *x: (print(*x), rep.append(" ".join(str(y) for y in x)))

    # ---------------- 1. rotazione dalle posizioni -----------------------------
    pr = pairs(ws, g)
    PA = np.array([[ws.x[n], ws.y[n], ws.z[n]] for n, _, _ in pr])
    PG = np.array([rg for _, _, rg in pr])
    labs = np.array([l for _, l, _ in pr])
    R_fit, t_fit = kabsch(PG, PA)
    res = np.linalg.norm(PA - (PG @ R_fit.T + t_fit), axis=1)
    P("== 1. ROTAZIONE GINO -> SHARPy ricavata dalle POSIZIONI dei nodi (Kabsch)")
    P("R_fit =\n", np.round(R_fit, 4))
    P("residuo posizioni: medio %.1f mm, max %.1f mm (%d nodi)" % (1e3 * res.mean(), 1e3 * res.max(), len(res)))
    for lab in ("rudder", "elevator_port", "elevator_stbd"):
        m = labs == lab
        P("   %-14s residuo max %.1f mm" % (lab, 1e3 * res[m].max()))
    P("|R_fit - R_attuale|max = %.2e   |R_fit - R_vecchia|max = %.2e"
      % (np.abs(R_fit - R_NEW).max(), np.abs(R_fit - R_OLD).max()))
    R_swap = None
    # stessa prova con l'accoppiamento dei rami invertito (port<->stbd)
    pr2 = []
    for (n, lab, rg) in pr:
        if lab != "rudder":
            rg = rg * np.array([1, 1, -1])
        pr2.append(rg)
    Rs, ts = kabsch(np.array(pr2), PA)
    res2 = np.linalg.norm(PA - (np.array(pr2) @ Rs.T + ts), axis=1)
    P("controprova rami scambiati port<->stbd: residuo max %.1f mm (deve essere grande)" % (1e3 * res2.max()))

    # ---------------- 3. normali dei pannelli ---------------------------------
    # zeta e' nel frame inerziale G di SHARPy; la trave (e i gdl del modello
    # lineare) nel frame A: si riporta zeta in A con il quaternione iniziale
    import sharpy.utils.algebra as algebra
    ts0 = data.structure.timestep_info[0]
    C_GA = algebra.quat2rotation(ts0.quat)
    P("C_GA (orientamento iniziale di A in G) =\n", np.round(C_GA, 4))
    zeta = [np.einsum("ji,jab->iab", C_GA, z - ts0.for_pos[:3, None, None])
            for z in data.aero.timestep_info[0].zeta]
    names = []
    P("\n== 3. NORMALI DEI PANNELLI (frame CAD; z_CAD verso l'alto, x_CAD verso prua)")
    normals = []
    for isurf, z in enumerate(zeta):
        d1 = z[:, 1:, :-1] - z[:, :-1, :-1]          # lungo la corda
        d2 = z[:, :-1, 1:] - z[:, :-1, :-1]          # lungo lo span
        nA = np.cross(d1, d2, axis=0).reshape(3, -1).T
        nA /= np.linalg.norm(nA, axis=1)[:, None]
        nC = avec_to_cad(nA)
        cen = a_to_cad(0.25 * (z[:, 1:, 1:] + z[:, :-1, :-1] + z[:, 1:, :-1] + z[:, :-1, 1:])
                       .reshape(3, -1).T, s0)
        normals.append((cen, nC))
        mean = nC.mean(0)
        dom = int(np.argmax(np.abs(mean)))
        ok = np.all(np.sign(nC[:, dom]) == np.sign(mean[dom]))
        nm_ = "timone" if np.ptp(cen[:, 2]) > 1.0 else ("elev y_CAD>0" if cen[:, 1].mean() > 0 else "elev y_CAD<0")
        P("   estensione CAD min %s max %s" % (np.round(cen.min(0), 3), np.round(cen.max(0), 3)))
        names.append(nm_)
        P("superficie %d (%s): %d pannelli, normale media CAD %s, componente dominante %s, "
          "tutte concordi: %s, min/max n_%s = %.3f/%.3f"
          % (isurf, nm_, len(nC), np.round(mean, 3), "xyz"[dom], ok, "xyz"[dom],
             nC[:, dom].min(), nC[:, dom].max()))

    # ---------------- 2. segno fisico, teoria delle strisce -------------------
    P("\n== 2. SEGNO FISICO: Re Q(k=0), strisce quasi-statiche con i modi di GINO")
    t_cad = np.array([-1.0, 0, 0])                    # corrente verso -x_CAD
    Rinv = R_fit.T
    Qs = np.zeros((NM, NM))
    for isurf, z in enumerate(zeta):
        lab_s = names[isurf]
        nst = z.shape[2] - 1
        for j in range(nst):
            le = 0.5 * (z[:, 0, j] + z[:, 0, j + 1]); te = 0.5 * (z[:, -1, j] + z[:, -1, j + 1])
            c = np.linalg.norm(te - le)
            ds = np.linalg.norm(z[:, 0, j + 1] - z[:, 0, j])
            ac_A = le + 0.25 * (te - le)
            n_C = normals[isurf][1].reshape(z.shape[1] - 1, nst, 3)[:, j].mean(0)
            n_C /= np.linalg.norm(n_C)
            # punto di GINO della stazione: posizione del quarto di corda in GINO
            ac_G = Rinv @ (ac_A - t_fit)
            if lab_s == "timone":
                lab, yc = "rudder", ac_G[1]
            else:
                lab = "elevator_port" if ac_G[2] > 0 else "elevator_stbd"
                yc = abs(ac_G[2])
            u, rg = interp_mode(g, lab, yc, NM)
            ac_C = gino_to_cad(ac_G[None])[0]; rg_C = gino_to_cad(rg[None])[0]
            uC = gvec_to_cad(u[:3].T); thC = gvec_to_cad(u[3:].T)
            uac = uC + np.cross(thC, ac_C - rg_C)
            # cabrata = bordo d'attacco verso +n = corda ruotata t + th x t con
            # componente NEGATIVA su n: alpha = -(th x t).n
            alpha = -np.cross(thC, t_cad) @ n_C           # (NM,)
            w = uac @ n_C
            Qs += 2 * np.pi * c * ds * np.outer(w, alpha)
    q = FSI / "FLUID" / "PanelAero" / "Qjj"
    Qn = np.load(q / ("Quvlm_AC38_TFOIL_water_n48c14e20_nm10%s.npz" % RS))["Q"][0, :NM, :NM].real
    Qo = np.load(q / "Quvlm_AC38_TFOIL_water_n48c14e20_nm6.npz")["Q"][0].real
    P("strisce (indipendente da SHARPy):\n", np.round(Qs, 4))
    P("UVLM nuova (R attuale):\n", np.round(Qn, 4))
    P("UVLM vecchia:\n", np.round(Qo, 4))
    big = np.abs(Qs) > 0.1 * np.abs(Qs).max()
    P("termini significativi (|Q|>10%% del max): %d; segno concorde con la NUOVA: %d, con la VECCHIA: %d"
      % (big.sum(), (np.sign(Qs) == np.sign(Qn))[big].sum(), (np.sign(Qs) == np.sign(Qo))[big].sum()))
    P("diag strisce", np.round(np.diag(Qs), 4))
    P("diag nuova  ", np.round(np.diag(Qn), 4))
    P("diag vecchia", np.round(np.diag(Qo), 4))
    (out / "report.txt").write_text("\n".join(rep))

    # ---------------- figure: modi trasferiti ----------------------------------
    views = [((0, 2), "vista laterale (x_CAD, z_CAD)"), ((1, 2), "vista da prua (y_CAD, z_CAD)"),
             ((0, 1), "vista dall'alto (x_CAD, y_CAD)")]
    nodes = np.array([n for n, _, _ in pr])
    for mode in range(NM):
        fig, ax = plt.subplots(1, 3, figsize=(18, 7))
        Ug = np.array([interp_mode(g, lab, (rg[1] if lab == "rudder" else abs(rg[2])), NM)[0][:, mode]
                       for _, lab, rg in pr])
        amp = 0.35 / max(np.abs(Ug[:, :3]).max(), 1e-12)
        baseC = gino_to_cad(PG)
        defG = baseC + amp * gvec_to_cad(Ug[:, :3])
        PA_c = a_to_cad(PA, s0)
        for R, col, name in ((R_NEW, "tab:blue", "SHARPy, R attuale"), (R_OLD, "tab:red", "SHARPy, R vecchia")):
            uA = Ug[:, :3] @ R.T
            thA = Ug[:, 3:] @ R.T
            defA = PA_c + amp * avec_to_cad(uA)
            # griglia dei pannelli deformata: trasporto rigido dal nodo piu' vicino
            for (iv, (i0, i1)) in enumerate([v[0] for v in views]):
                ax[iv].plot(defA[:, i0], defA[:, i1], "o", ms=3, color=col, label=name, alpha=.8)
            if R is R_NEW:
                for z in zeta:
                    pts = z.reshape(3, -1).T
                    near = np.argmin(np.linalg.norm(pts[:, None, :] - PA[None, ::3, :], axis=2), axis=1) * 3
                    dp = uA[near] + np.cross(thA[near], pts - PA[near])
                    pc = a_to_cad(pts + amp * dp, s0)
                    for iv, ((i0, i1), _) in enumerate(views):
                        ax[iv].plot(pc[:, i0], pc[:, i1], ",", color="tab:cyan", alpha=.5)
        for iv, ((i0, i1), tt) in enumerate(views):
            ax[iv].plot(baseC[:, i0], baseC[:, i1], ".", color="0.7", ms=2, label="indeformato")
            ax[iv].plot(defG[:, i0], defG[:, i1], "k-", lw=1.5, label="GINO (riferimento)")
            ax[iv].set_title(tt); ax[iv].set_aspect("equal"); ax[iv].grid(alpha=.3)
            ax[iv].set_xlabel("xyz"[i0] + "_CAD [m]"); ax[iv].set_ylabel("xyz"[i1] + "_CAD [m]")
        ax[0].legend(loc="lower left", fontsize=8)
        fig.suptitle("Modo %d (%.2f Hz): GINO vs trasferito su SHARPy  -  "
                     "il blu deve stare sul nero, il rosso e' la R vecchia; ciano = pannelli deformati (R attuale)"
                     % (mode + 1, g["f"][mode]))
        fig.tight_layout(); fig.savefig(out / ("modo_%02d.png" % (mode + 1)), dpi=110); plt.close(fig)

    # ---------------- figura: normali -------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    axx = fig.add_subplot(111, projection="3d")
    for (cen, nC), nm_ in zip(normals, names):
        st = max(1, len(cen) // 250)
        axx.quiver(cen[::st, 0], cen[::st, 1], cen[::st, 2], nC[::st, 0], nC[::st, 1], nC[::st, 2],
                   length=0.08, normalize=True, color={"timone": "tab:blue"}.get(nm_, "tab:red"))
        axx.plot(cen[:, 0], cen[:, 1], cen[:, 2], ",", color="0.5")
    axx.set_xlabel("x_CAD"); axx.set_ylabel("y_CAD"); axx.set_zlabel("z_CAD")
    axx.set_box_aspect((np.ptp(PA_c[:, 0]) + .3, np.ptp(PA_c[:, 1]) + .3, np.ptp(PA_c[:, 2])))
    axx.set_title("Normali dei pannelli UVLM (blu timone, rosso elevatore), frame CAD")
    fig.savefig(out / "normali_pannelli.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for (cen, nC), nm_ in zip(normals, names):
        if nm_ == "timone":
            ax[0].scatter(cen[:, 0], cen[:, 2], c=nC[:, 1], cmap="coolwarm", vmin=-1, vmax=1, s=4)
        else:
            ax[1].scatter(cen[:, 0], cen[:, 1], c=nC[:, 2], cmap="coolwarm", vmin=-1, vmax=1, s=4)
    ax[0].set_title("timone: n_y CAD per pannello (tutto un colore = coerente)")
    ax[1].set_title("elevatore: n_z CAD per pannello (rosso = verso l'alto)")
    for a_ in ax: a_.set_aspect("equal"); a_.grid(alpha=.3)
    fig.savefig(out / "normali_mappa.png", dpi=110); plt.close(fig)
    print("figure e report in", out)


if __name__ == "__main__":
    main()
