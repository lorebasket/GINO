#!/usr/bin/env python3
"""Contributi nodali delle GAF UVLM del T AC38: da dove nasce ogni Q_ij (lungo la lama e sulle semiali).

Stessa catena di precompute_quvlm_tfoil.py (build, gino_to_sharpy), ma l'uscita non viene proiettata subito:
si tengono le forze nodali F(omega) per ciascun modo forzato j e il contributo del nodo n a Q_ij e'
sum_{gdl di n} Phi[gdl, i] F[gdl, j]. La somma sui nodi riproduce la GAF della tabella.
Convenzione dell'uscita: DT (F = rho U^2 Q q, Im < 0 smorzante, k_bar = omega 0.15/U), modi nell'ordine e coi
segni DT (MAC >= 0.96, vedi examples/AC38_TFOIL/confronto_gaf_urans_dt.py).

    python gino_bem/gaf_contributi_tfoil.py --gino <dry npz> --out <npz> [--kbar 0,0.3,0.6,1.0,1.16]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SHARPY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SHARPY_ROOT))
from gino_bem.precompute_quvlm_tfoil import build, gino_to_sharpy, KN   # noqa: E402

PERM = [0, 1, 2, 3, 4, 6, 5, 7]
SGN = np.array([1, 1, -1, -1, 1, -1, 1, -1], float)
B_REF = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gino", type=Path, required=True)
    ap.add_argument("--grid", default="n48c14e20")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kbar", default="0,0.3,0.6,1.0,1.16")
    a = ap.parse_args()
    import sharpy.linear.src.libss as libss
    nm = 8
    g = dict(np.load(a.gino, allow_pickle=True))
    tag = "tfoil_contrib" + (("_img" + os.environ["AC38_IMAGE"]) if os.environ.get("AC38_IMAGE") else "")
    ws, dn = build(a.grid, False, nm, tag)
    Phi = gino_to_sharpy(ws, dn, g)[:, :nm]
    Phi_o = Phi * gino_to_sharpy.out_mask[:, None]
    V = 50.0 * KN
    ls = dn.linear.linear_system
    ls.update(float(V))
    ss = ls.uvlm.ss
    facts = ss.ScalingFacts if hasattr(ss, 'ScalingFacts') else ls.uvlm.sys.ScalingFacts
    F_ref = float(facts["dyn_pressure"]) * float(facts["length"]) ** 2
    t_ref = float(facts["length"]) / float(V)
    dimfac = F_ref / float(facts["dyn_pressure"])
    nd = Phi.shape[0]
    Bin = np.zeros((ss.inputs, 2 * nm)); Bin[:nd, :nm] = Phi; Bin[nd:2 * nd, nm:] = Phi
    Cn = np.zeros((nd, ss.outputs)); Cn[:, :nd] = np.eye(nd)
    red = libss.StateSpace(ss.A, ss.B @ Bin, Cn @ ss.C, Cn @ ss.D @ Bin, dt=ss.dt)
    kbar = np.array([float(x) for x in a.kbar.split(",")])
    k_pa = kbar / B_REF
    w_s = k_pa * V * t_ref
    H = red.freqresp(w_s)                                         # (nd, 2 nm, nk)
    # forze nodali come nella tabella (conj = segno grezzo DLM), poi conj del p-k -> convenzione DT
    Fn = np.array([dimfac * (H[:, :nm, i] + 1j * w * H[:, nm:, i]) for i, w in enumerate(w_s)])   # (nk, nd, nm)
    vdof = np.asarray(dn.structure.vdof)
    pos = dn.structure.timestep_info[0].pos
    nodes = [n for n in range(len(vdof)) if vdof[n] >= 0]
    lab = {}
    for name, key in (("elevator_stbd", "port"), ("elevator_port", "stbd")):
        for n in ws.elev_nodes[key][1:]:
            lab[int(n)] = name
    img = set(getattr(ws, "image_map", {}) or {})
    contrib = np.zeros((len(kbar), len(nodes), nm, nm), complex)
    cad = np.zeros((len(nodes), 3)); br = []
    S = np.outer(SGN, SGN)
    for c, n in enumerate(nodes):
        j = int(vdof[n]); sl = slice(6 * j, 6 * j + 6)
        for i in range(len(kbar)):
            q = Phi_o[sl].T @ Fn[i, sl, :]                        # (nm, nm) contributo GINO, segno grezzo
            contrib[i, c] = 0.5 * S * q[np.ix_(PERM, PERM)]       # conj(conj(.)) = . ; /2 -> DT
        p = pos[n]
        cad[c] = [0.777 - p[0], p[1], 0.565 - (p[2] - ws.z[0])]
        br.append("image" if n in img else lab.get(n, "rudder"))
    Qtot = contrib.sum(1)
    np.savez(a.out, kbar=kbar, U=V, contrib=contrib, Q=Qtot, node_cad=cad, branch=np.array(br),
             note="contributi nodali a Q_ij (convenzione DT, modi DT 1..8); Q = somma sui nodi")
    print("Q44 per k_bar", np.round(Qtot[:, 3, 3], 4))
    print("salvato", a.out)


if __name__ == "__main__":
    main()
