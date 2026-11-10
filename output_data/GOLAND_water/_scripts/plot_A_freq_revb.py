#!/usr/bin/env python3
"""Rev B (21/10/2026): massa aggiunta BEM A_ii(f; h/c) del caso numerical A, modi 1 e 2, con la massa
non circolatoria dell'UVLM (fit della tabella GAF, k in [14, 24], come nel p-k delta) per confronto.
Sostituisce fig_A_vs_frequenza.png (sonda v2, struttura diversa).
    python output_data/GOLAND_water/_scripts/plot_A_freq_revb.py"""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, NullFormatter
XT = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]

GW = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FSI = os.path.abspath(os.path.join(GW, "..", ".."))
RHO = 997.0
REV = os.environ.get("GW_REV", "revA")   # revD: BEM con torsione corretta
P = {}
for d in sorted(glob.glob(os.path.join(GW, "casoA_numeric_" + REV, "*_delta_*_hc*"))):
    hc = float(d.rsplit("_hc", 1)[1].replace("p", "."))
    z = np.load(glob.glob(os.path.join(d, "capytaine", "modal_radiation_AB_depth_*.npz"))[0], allow_pickle=True)
    P[hc] = (np.asarray(z["omega"], float), np.asarray(z["added_mass"], float))
    if abs(hc - 1.0) < 1e-9:
        log = open(os.path.join(d, "flutter.log"), errors="ignore").read()
        tb = np.load(os.path.join(FSI, re.search(r"modal GAF table (\S+\.npz)", log).group(1)))
        kl, Q = tb["k_list"], tb["Q"]; m = (kl >= 14) & (kl <= 24)
        ANC = np.array([0.5 * RHO * np.polyfit(kl[m] ** 2, Q[m, i, i].real, 1)[0] for i in range(Q.shape[1])])
hcs = sorted(P); deep = hcs[-1]
cols = dict(zip(hcs, cm.viridis(np.linspace(0, .9, len(hcs)))))
FF = (2.1, 2.4) if REV == "revA" else (2.75, 2.88)   # frequenze di flutter del caso numerical A

fig, ax = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)
for j, (mi, nm) in enumerate([(0, "modo 1 (flessione)"), (1, "modo 2 (flesso-torsionale, ramo di flutter)")]):
    wd, Ad = P[deep]
    for hc in hcs:
        w, A = P[hc]; f = w / (2 * np.pi)
        ax[0, j].plot(f, A[:, mi, mi], color=cols[hc], lw=2, label=f"BEM h/c = {hc:g}")
        ax[1, j].plot(f, 100 * (A[:, mi, mi] / np.interp(w, wd, Ad[:, mi, mi]) - 1), color=cols[hc], lw=2)
    ax[0, j].axhline(ANC[mi], color="crimson", ls="--", lw=2, label="UVLM (non vede h)")
    du = 100 * (ANC[mi] / Ad[-1, mi, mi] - 1)
    if abs(du) < 95:
        ax[1, j].axhline(du, color="crimson", ls="--", lw=2)
    else:
        ax[1, j].annotate(f"UVLM: {du:+.0f} % (fuori scala)", (0.07, 0.92), xycoords="axes fraction", color="crimson", fontsize=10, fontweight="bold")
    for a in ax[:, j]:
        a.axvspan(*FF, color="orange", alpha=.25); a.grid(alpha=.3, which="both"); a.set_xscale("log"); a.set_xticks(XT); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a.xaxis.set_minor_formatter(NullFormatter())
    ax[0, j].set_title("A$_{%d%d}$ – %s" % (mi + 1, mi + 1, nm), fontsize=11)
    ax[1, j].axhline(0, color="k", lw=1); ax[1, j].set_ylim(-95, 95)
    ax[1, j].set_xlabel("frequenza di oscillazione f [Hz]")
    ax[0, j].set_ylim(0, None)
ax[0, 0].set_ylabel("massa aggiunta modale [–]"); ax[1, 0].set_ylabel("ΔA rispetto a h/c = %g [%%]" % deep)
ax[0, 1].legend(fontsize=8.5, ncol=2, loc="upper right")
ax[1, 0].annotate("banda di flutter\n(%.2f–%.2f Hz)" % FF, (sum(FF) / 2, 70), ha="center", fontsize=9, color="#a05a00")
ax[1, 1].annotate("transitorio a bassa f (parete rigida: A cresce)", (0.03, 84), ha="center", fontsize=9)
ax[1, 1].annotate("alta f: superficie a pressione nulla,\nA resta ridotta e dipende solo da h/c", (9, 35), ha="center", fontsize=9)
fig.suptitle("Caso numerical A – massa aggiunta BEM al variare di frequenza e sommergenza, confronto con l'UVLM",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
out = os.path.join(GW, "figures", "fig_A_freq_sommergenza_revb.png" if REV == "revA" else "fig_A_freq_sommergenza_%s.png" % REV)
fig.savefig(out, dpi=150); print(out)
for hc in hcs:
    w, A = P[hc]; i = np.argmin(abs(w - 2 * np.pi * 20))
    print("h/c %4g  A11 %.3f  A22 %.3f  (20 Hz)  dA22 vs deep %+.1f %%" % (hc, A[i, 0, 0], A[i, 1, 1], 100 * (A[i, 1, 1] / P[deep][1][i, 1, 1] - 1)))
print("UVLM diag", ANC)
