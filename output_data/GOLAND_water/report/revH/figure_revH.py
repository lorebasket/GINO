"""Figure del report GOLAND_water rev H (10/11/2026, per Pietro): come rev G, con legende senza riferimenti interni.
Nessun calcolo nuovo: legge le scansioni di rev E (UVLM in fluido infinito + BEM) e del test P.

    python figure_revH.py            # scrive PNG e dati_revG.json accanto a questo file
"""
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GW = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("figure_revE", HERE.parent / "revE" / "figure_revE.py")
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

G, C0, KN, BAND = 9.81, 0.294, 0.5144, (20, 40)
HC = fe.HC
COL, INK, MUTED, GRID = fe.COL, fe.INK, fe.MUTED, fe.GRID
P = json.load(open(GW / "testP" / "risultati_testP.json"))
O = {r["hc"]: r for r in json.load(open(GW / "testO" / "risultati_testO.json"))["rows"]}
runs = fe.runs_of(1)
scan_img = lambda hc: GW / "testP" / ("scan_img_hc%s.txt" % ("%.1f" % hc).replace(".", "p"))


def fig_schema():
    """ala vera a -h, ala immagine a +h con la stessa circolazione; superficie a pressione nulla"""
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    xs = np.linspace(0, 1, 200)
    yt = 5 * 0.12 * (0.2969 * np.sqrt(xs) - 0.126 * xs - 0.3516 * xs ** 2 + 0.2843 * xs ** 3 - 0.1015 * xs ** 4)
    h = 0.6
    ax.axhspan(-1.25, 0, color="#2E6FB7", alpha=0.10, lw=0)
    ax.plot([-0.9, 2.9], [0, 0], color="#2E6FB7", lw=2.5)
    ax.text(2.85, 0.05, "superficie libera: φ = 0", color="#2E6FB7", ha="right", va="bottom", fontsize=11)
    for z, col, lab in [(-h, INK, "ala vera"), (+h, "#8E6BB8", "ala immagine (copia traslata a +2h)")]:
        ax.fill_between(xs, z - yt, z + yt, color=col, alpha=1.0 if col == INK else 0.55, lw=0)
        ax.annotate(lab, (1.0, z + 0.13), color=col, fontsize=11, va="bottom")
        # circolazione: stesso verso (orario, portanza verso l'alto con flusso da sinistra)
        t = np.linspace(0.25 * np.pi, 1.85 * np.pi, 60)
        ax.plot(0.25 + 0.17 * np.cos(-t), z + 0.17 * np.sin(-t), color=col, lw=1.6)
        ax.annotate("", xy=(0.25 + 0.17 * np.cos(-t[-1]), z + 0.17 * np.sin(-t[-1])),
                    xytext=(0.25 + 0.17 * np.cos(-t[-4]), z + 0.17 * np.sin(-t[-4])),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.annotate("Γ", (0.25, z + 0.24), color=col, fontsize=12, ha="center")
        # scia
        ax.plot([1.0, 2.0], [z, z], color=col, lw=1, ls=(0, (4, 3)))
    ax.annotate("", xy=(-0.3, 0), xytext=(-0.3, -h), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(-0.35, -h / 2, "h", color=MUTED, ha="right", va="center", fontsize=12)
    ax.annotate("", xy=(-0.3, h), xytext=(-0.3, 0), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(-0.35, h / 2, "h", color=MUTED, ha="right", va="center", fontsize=12)
    ax.annotate("U", (-0.85, -0.95), color=INK, fontsize=12)
    ax.annotate("", xy=(-0.45, -0.92), xytext=(-0.78, -0.92), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text(2.85, -1.18, "w indotta simmetrica, u antisimmetrica rispetto a z = 0:\nl'immagine toglie downwash all'ala vera (effetto biplano)",
            color=MUTED, ha="right", va="bottom", fontsize=10)
    ax.set_xlim(-0.9, 2.9); ax.set_ylim(-1.25, 1.0)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(HERE / "schema_immagine.png"); plt.close(fig)


def fig_confronto():
    """V_f contro h/c: nessuna superficie, superficie solo nel BEM (rev F), anche nell'UVLM (rev G)"""
    r = P["rows"]
    hc = np.array([x["hc"] for x in r])
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    ax.axhline(P["uvlm"], color=INK, lw=1.5, ls="--", label="nessuna superficie libera (%.2f kn)" % P["uvlm"])
    ax.plot(hc, [x["v0"] for x in r], "o-", color="#2E6FB7", lw=2, ms=7, mec="white", mew=1.5, label="superficie solo nella massa aggiunta (BEM)")
    ax.plot(hc, [x["vp"] for x in r], "o-", color="#E0622B", lw=2.4, ms=8, mec="white", mew=1.5, label="superficie anche nelle forze circolatorie (UVLM con immagine)")
    for x in r:
        ax.annotate("%.2f" % x["vp"], (x["hc"], x["vp"]), (6, -14 if x["hc"] >= 1 else 6), textcoords="offset points", color="#E0622B", fontsize=9)
    ax.set_xlabel("sommergenza h/c"); ax.set_ylabel("V_f [kn]")
    ax.set_xlim(2.1, 0.3); ax.set_ylim(41.5, 47)
    ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(HERE / "confronto_vf.png"); plt.close(fig)


def fig_sigma():
    """sigma/omega del ramo di flutter: continua = rev F, tratteggiata = rev G"""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.axhline(0, color=MUTED, lw=1)
    for hc in HC:
        for src, ls in [(runs[hc], "-"), (scan_img(hc), "--")]:
            V, s, f = fe.envelope(src, BAND)
            ax.plot(V, s / (2 * np.pi * f), ls, color=COL[hc], lw=2)
            vf, _ = fe.crossing(V, s, f)
            ax.plot([vf], [0], "o", color=COL[hc], ms=7, mec="white", mew=1.5, zorder=4)
    for hc in HC:
        ax.plot([], [], color=COL[hc], lw=2.5, label="h/c %g" % hc)
    ax.plot([], [], "-", color=INK, lw=1.5, label="superficie solo nella massa aggiunta (BEM)")
    ax.plot([], [], "--", color=INK, lw=1.5, label="anche nelle forze circolatorie (immagine)")
    ax.set_xlim(38, 50); ax.set_ylim(-0.06, 0.2); ax.set_xlabel("V [kn]"); ax.set_ylabel("σ / ω del ramo di flutter  [−]")
    ax.grid(True, color=GRID, lw=0.8); ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(HERE / "confronto_sigma.png"); plt.close(fig)


if __name__ == "__main__":
    fig_schema()
    fig_confronto()
    fig_sigma()
    rows = []
    for x in P["rows"]:
        U, w, h = x["vp"] * KN, 2 * math.pi * x["fp"], x["hc"] * C0
        rows.append(dict(x, fnh=U / math.sqrt(G * h), w2h=w * w * h / G, tau=U * w / G, f3d=O[x["hc"]]["f3d"]))
    for r in rows:
        print({k: round(v, 3) for k, v in r.items()})
    json.dump({"uvlm": P["uvlm"], "rows": rows}, open(HERE / "dati_revG.json", "w"), indent=1)
