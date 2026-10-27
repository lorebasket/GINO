"""Analisi della divergenza statica del T AC38 (modello 5: UVLM + BEM, 10 modi).

  conda run -n sharpy python sharpy/gino_bem/analyze_tfoil_divergence.py

La divergenza e' il problema statico K u = q Re Q(0) u: il BEM non entra (la
massa aggiunta moltiplica omega^2 = 0) e la rimozione della massa apparente
dell'UVLM e' nulla a k = 0. Si calcolano:
  1. velocita' e modo di divergenza (base dei modi secchi);
  2. forma fisica nel frame CAD: spostamenti e angolo d'incidenza locale
     lungo il timone e lungo l'elevatore;
  3. lavoro delle forze idrodinamiche quasi-statiche, striscia per striscia
     (modello a strisce 2*pi, che concorda col segno della GAF UVLM), separato
     fra timone ed elevatore: chi destabilizza;
  4. leva del carico dell'elevatore rispetto alla punta della lama.
Uscite in output_data/AC38_TFOIL/12_divergenza/.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eig
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sharpy.utils.algebra as algebra
from gino_bem.check_tfoil_projection import (build_geom, a_to_cad, avec_to_cad, gino_to_cad,
                                             gvec_to_cad, interp_mode, kabsch, pairs)

import os
FSI = Path(__file__).resolve().parents[3]
RS = "" if os.environ.get("AC38_REV", "reva") == "legacy" else "_reva"   # revisione del modello (vedi config)
OUT = FSI / "output_data" / "AC38_TFOIL" / ("12_divergenza" + RS); OUT.mkdir(parents=True, exist_ok=True)
NM, RHO, KN = 10, 997.0, 1.94384
rep = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); rep.append(s)

g = dict(np.load(FSI / ("output_data/AC38_TFOIL/00_dry%s/tfoil_dry_modes.npz" % RS), allow_pickle=True))
f = g["f"][:NM]; K = np.diag((2 * np.pi * f) ** 2)
Q0 = np.real(np.load(FSI / ("FLUID/PanelAero/Qjj/Quvlm_AC38_TFOIL_water_n48c14e20_nm10%s.npz" % RS))["Q"][0])

# ---------------------------------------------------------------- 1. modo
w, V = eig(K, Q0)
ok = np.isfinite(w) & (np.abs(w.imag) < 1e-6 * np.abs(w)) & (w.real > 0)
order = np.flatnonzero(ok)[np.argsort(w.real[ok])]
P("== 1. RADICI STATICHE (K u = q Re Q(0) u)")
for i in order[:3]:
    v = np.real(V[:, i]); v /= v[np.argmax(np.abs(v))]
    P("   V = %.1f kn (%.2f m/s), q = %.3g Pa, ampiezze modali %s"
      % (np.sqrt(2 * w[i].real / RHO) * KN, np.sqrt(2 * w[i].real / RHO), w[i].real, np.round(v, 3)))
i0 = order[0]; v = np.real(V[:, i0]); v /= v[np.argmax(np.abs(v))]
qdiv = w[i0].real

# divergenza con sottoinsiemi di modi: chi serve per averla
P("\n   sottoinsiemi di modi (V_div in kn):")
for S in ([1], [1, 7], [1, 5], [1, 5, 7], [5, 7], [7], [0], list(range(NM))):
    ww, _ = eig(K[np.ix_(S, S)], Q0[np.ix_(S, S)])
    o = np.isfinite(ww) & (np.abs(ww.imag) < 1e-6 * np.abs(ww)) & (ww.real > 0)
    P("     modi %-22s -> %s" % ([s + 1 for s in S],
      "%.1f" % (np.sqrt(2 * ww.real[o].min() / RHO) * KN) if o.any() else "nessuna"))

# ---------------------------------------------------------------- geometria
ws, data = build_geom("n48c14e20")
ts0 = data.structure.timestep_info[0]
C_GA = algebra.quat2rotation(ts0.quat); s0 = ws.z[0]
zeta = [np.einsum("ji,jab->iab", C_GA, z - ts0.for_pos[:3, None, None]) for z in data.aero.timestep_info[0].zeta]
pr = pairs(ws, g)
PA = np.array([[ws.x[n], ws.y[n], ws.z[n]] for n, _, _ in pr]); PG = np.array([rg for _, _, rg in pr])
R_fit, t_fit = kabsch(PG, PA)

# modo fisico: combinazione dei modi secchi
Phi = g["Phi"][:, :NM] @ v
gm = dict(g); gm["Phi"] = Phi[:, None]

t_cad = np.array([-1.0, 0, 0])
strips = []      # (superficie, coord span, c, ds, alpha [rad per ampiezza], w_n, lavoro, x_AC, punto)
for isurf, z in enumerate(zeta):
    nst = z.shape[2] - 1
    for j in range(nst):
        le = 0.5 * (z[:, 0, j] + z[:, 0, j + 1]); te = 0.5 * (z[:, -1, j] + z[:, -1, j + 1])
        c = np.linalg.norm(te - le); ds = np.linalg.norm(z[:, 0, j + 1] - z[:, 0, j])
        ac_A = le + 0.25 * (te - le)
        d1 = te - le; d2 = z[:, 0, j + 1] - z[:, 0, j]
        nA = np.cross(d1, d2); nA /= np.linalg.norm(nA); n_C = avec_to_cad(nA[None])[0]
        ac_G = R_fit.T @ (ac_A - t_fit)
        if isurf == 0:
            lab, yc = "rudder", ac_G[1]
        else:
            lab = "elevator_port" if ac_G[2] > 0 else "elevator_stbd"; yc = abs(ac_G[2])
        u, rg = interp_mode(gm, lab, yc, 1)
        ac_C = gino_to_cad(ac_G[None])[0]; rg_C = gino_to_cad(rg[None])[0]
        uC = gvec_to_cad(u[:3].T)[0]; thC = gvec_to_cad(u[3:].T)[0]
        uac = uC + np.cross(thC, ac_C - rg_C)
        alpha = -np.cross(thC, t_cad) @ n_C
        wn = uac @ n_C
        work = 2 * np.pi * c * ds * alpha * wn          # contributo a v^T Q_strisce v
        strips.append((isurf, ac_C, c, ds, alpha, wn, work, thC, uac))

W = np.array([s[6] for s in strips]); surf = np.array([s[0] for s in strips])
Qs_vv = W.sum()
P("\n== 3. LAVORO QUASI-STATICO SUL MODO DI DIVERGENZA (strisce, v^T Q v)")
P("   UVLM  v^T Q0 v = %.4g      strisce = %.4g" % (v @ Q0 @ v, Qs_vv))
for k, nm in ((0, "timone"), (1, "elevatore y>0"), (2, "elevatore y<0")):
    P("   %-15s %+.4g  (%.0f %%)" % (nm, W[surf == k].sum(), 100 * W[surf == k].sum() / Qs_vv))
P("   rigidezza strutturale v^T K v = %.4g ; q_div = v^T K v / v^T Q v = %.3g Pa (UVLM %.3g)"
  % (v @ K @ v, (v @ K @ v) / (v @ Q0 @ v), qdiv))

# ---------------------------------------------------------------- 2/4. forma e leva
tipC = gino_to_cad(PG[[np.argmax(PG[:, 1] * (np.array([l for _, l, _ in pr]) == "rudder"))]])[0]
# punta della lama in GINO = nodo rudder con y massimo
rud = g["sub"] == "rudder"; it = np.flatnonzero(rud)[np.argmax(g["pos"][rud, 1])]
U = Phi.reshape(len(g["pos"]), 6)
tip_u = gvec_to_cad(U[it, :3][None])[0]; tip_th = gvec_to_cad(U[it, 3:][None])[0]
tip_C = gino_to_cad(g["pos"][it][None])[0]
P("\n== 2. FORMA FISICA (frame CAD, ampiezza normalizzata)")
P("   punta lama (%.3f, %.3f, %.3f): u = %s mm/unita', rotazione = %s mrad/unita'"
  % (*tip_C, np.round(1e3 * tip_u, 3), np.round(1e3 * tip_th, 3)))
el = [s for s in strips if s[0] > 0]
aC = np.array([s[1] for s in el]); Wl = np.array([s[6] for s in el])
# incidenza con verso fisico: positiva = cabrante (portanza verso +z_CAD). Le due
# semiali hanno normali opposte, quindi alpha (riferito a n) va riportato a +z.
n_up = []
for isurf, z in enumerate(zeta):
    if isurf == 0: continue
    for j in range(z.shape[2] - 1):
        d1 = 0.5 * ((z[:, -1, j] + z[:, -1, j + 1]) - (z[:, 0, j] + z[:, 0, j + 1])); d2 = z[:, 0, j + 1] - z[:, 0, j]
        n_up.append(np.sign(avec_to_cad(np.cross(d1, d2)[None])[0][2]))
al = np.array([s[4] for s in el]) * np.array(n_up)
# parte rigida: beccheggio della punta della lama, theta x t proiettato su +z
al_rigid = -np.cross(tip_th, t_cad) @ np.array([0, 0, 1.0])
L = 2 * np.pi * np.array([s[2] * s[3] for s in el]) * al          # portanza / q (con segno, lungo n)
nz = np.array([np.sign(s[8] @ np.array([0, 0, 1.0])) for s in el])
xac = (L * aC[:, 0]).sum() / L.sum()
P("   elevatore: incidenza (cabrante +) media %.1f mrad/unita', min %.1f, max %.1f; portanza risultante a x_CAD = %.3f m"
  % (1e3 * al.mean(), 1e3 * al.min(), 1e3 * al.max(), xac))
P("   di cui beccheggio rigido dato dalla punta della lama: %.1f mrad; torsione propria dell'elevatore: media %.1f mrad"
  % (1e3 * al_rigid, 1e3 * (al.mean() - al_rigid)))
# lavoro: separa la parte dovuta al beccheggio rigido
P("   segno del lavoro dell'elevatore: incidenza e spostamento normale concordi (destabilizza) su %d strisce su %d"
  % (int((Wl > 0).sum()), len(Wl)))
P("   punta lama x_CAD = %.3f m  ->  la portanza dell'elevatore agisce %.0f mm %s della punta"
  % (tip_C[0], 1e3 * abs(xac - tip_C[0]), "DAVANTI" if xac > tip_C[0] else "DIETRO"))
ru = [s for s in strips if s[0] == 0]
P("   timone: incidenza locale max |alpha| = %.3f mrad/unita' (il timone quasi non lavora)"
  % (1e3 * np.abs([s[4] for s in ru]).max()))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# forma del timone: spostamenti x e y lungo z
pr_r = g["pos"][rud]; o = np.argsort(pr_r[:, 1]); Ur = U[rud][o]
zC = gino_to_cad(pr_r[o])[:, 2]; uC = gvec_to_cad(Ur[:, :3]); thC = gvec_to_cad(Ur[:, 3:])
ax[0, 0].plot(1e3 * uC[:, 0], zC, label="u_x (avanti-indietro)"); ax[0, 0].plot(1e3 * uC[:, 1], zC, label="u_y (laterale)")
ax[0, 0].plot(1e3 * thC[:, 1], zC, "--", label="rotazione attorno a y (beccheggio) [mrad]")
ax[0, 0].plot(1e3 * thC[:, 2], zC, ":", label="torsione attorno a z [mrad]")
ax[0, 0].set_title("Timone: modo di divergenza (%.1f kn)" % (np.sqrt(2 * qdiv / RHO) * KN)); ax[0, 0].set_ylabel("z_CAD [m]")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
yy = aC[:, 1]; oo = np.argsort(yy)
ax[0, 1].plot(yy[oo], 1e3 * al[oo], "o-", ms=3, label="incidenza locale, cabrante + [mrad]")
th_root = 1e3 * tip_th[1]
ax[0, 1].axhline(1e3 * al_rigid, color="k", ls="--", lw=.8, label="beccheggio rigido dalla punta della lama")
ax[0, 1].set_title("Elevatore: incidenza lungo l'apertura"); ax[0, 1].set_xlabel("y_CAD [m]"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3)
zr = np.array([s[1][2] for s in ru]); Wr = np.array([s[6] for s in ru])
ax[1, 0].barh(zr, Wr, height=0.05, color="tab:blue"); ax[1, 0].set_title("Lavoro destabilizzante per striscia – timone")
ax[1, 0].set_ylabel("z_CAD [m]"); ax[1, 0].grid(alpha=.3)
ax[1, 1].bar(yy, Wl, width=0.035, color="tab:red"); ax[1, 1].set_title("Lavoro destabilizzante per striscia – elevatore")
ax[1, 1].set_xlabel("y_CAD [m]"); ax[1, 1].grid(alpha=.3)
for a in ax[1]: a.axvline(0, color="k", lw=.6) if a is ax[1, 0] else a.axhline(0, color="k", lw=.6)
fig.tight_layout(); fig.savefig(OUT / "divergenza_modo_lavoro.png", dpi=120); plt.close(fig)

# vista laterale deformata
fig, ax = plt.subplots(figsize=(7, 9))
amp = 0.25 / max(np.abs(U[:, :3]).max(), 1e-12)
for lab, col in (("rudder", "k"), ("elevator_port", "tab:red"), ("elevator_stbd", "tab:red")):
    m = g["sub"] == lab; o = np.argsort(g["pos"][m, 1] if lab == "rudder" else np.abs(g["pos"][m, 2]))
    B = gino_to_cad(g["pos"][m][o]); D = B + amp * gvec_to_cad(U[m][o][:, :3])
    ax.plot(B[:, 0], B[:, 2], ":", color="0.6"); ax.plot(D[:, 0], D[:, 2], "-", color=col, lw=2)
for z in zeta[1:]:
    ax.plot(a_to_cad(z.reshape(3, -1).T, s0)[:, 0], a_to_cad(z.reshape(3, -1).T, s0)[:, 2], ",", color="tab:orange")
# freccia nel VERSO della portanza risultante (segno dell'incidenza cabrante media),
# lunghezza fissa: indica verso e punto d'applicazione, non il modulo
up = np.sign(L @ nz) if False else np.sign(al.mean())
ax.annotate("", xy=(xac, -2.75 + 0.35 * up), xytext=(xac, -2.75),
            arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2.5))
ax.text(xac + 0.02, -2.75 + 0.2 * up, "portanza\nelevatore\n(%s)" % ("verso l'alto" if up > 0 else "verso il basso"),
        color="tab:green", fontsize=9, va="center")
ax.axvline(tip_C[0], color="k", lw=.5, ls="--")
ax.set_aspect("equal"); ax.set_xlabel("x_CAD [m] (prua →)"); ax.set_ylabel("z_CAD [m]")
ax.set_title("Modo di divergenza, vista laterale (deformazione amplificata)"); ax.grid(alpha=.3)
fig.savefig(OUT / "divergenza_vista_laterale.png", dpi=120); plt.close(fig)
(OUT / "report.txt").write_text("\n".join(rep)); print("uscite in", OUT)
