"""Report rev H (copia di testP/grafici_vgvf.py con legende per Pietro): diagrammi V-g e V-f (tutte le radici della scansione auto-consistente) con e senza ala immagine,
per E3 e per il caso B. Scrive vgvf_E3.png e vgvf_casoB.png accanto a questo file.

    python grafici_vgvf.py
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
GW = HERE.parents[1]
TP = GW / "testP"
HC = [2.0, 1.0, 0.5, 0.4]
tag = lambda hc: ("%.1f" % hc).replace(".", "p")
INK, MUTED, GRID, IMG = "#222222", "#9A9A9A", "#E3E3E3", "#E0622B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(files):
    a = [np.loadtxt(f) for f in files if Path(f).exists()]
    return np.vstack(a) if a else np.zeros((0, 3))


def first_cross(a, band):
    """primo attraversamento sigma = 0 dell'inviluppo del ramo nella banda"""
    a = a[(a[:, 1] > band[0]) & (a[:, 1] < band[1])]
    V = np.unique(a[:, 0])
    s = np.array([a[a[:, 0] == v, 2].max() for v in V])
    for i in range(1, len(V)):
        if s[i - 1] < 0 <= s[i]:
            return V[i - 1] - s[i - 1] * (V[i] - V[i - 1]) / (s[i] - s[i - 1])
    return None


def figura(name, title, senza, con, band, xlim, flim, glim):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), dpi=170, sharex=True, sharey="row")
    for j, hc in enumerate(HC):
        for a, col, lab, z in [(senza[hc], MUTED, "senza immagine", 2), (con[hc], IMG, "con immagine", 3)]:
            if not len(a):
                continue
            g = a[:, 2] / (2 * np.pi * a[:, 1])
            axs[0, j].plot(a[:, 0], g, ".", ms=2.2, color=col, label=lab, zorder=z)
            axs[1, j].plot(a[:, 0], a[:, 1], ".", ms=2.2, color=col, zorder=z)
            vf = first_cross(a, band)
            if vf:
                for ax in axs[:, j]:
                    ax.axvline(vf, color=col, lw=1, ls="--", zorder=1)
                axs[0, j].annotate("%.2f kn" % vf, (vf, glim[1]), (3, -12 if col == IMG else -26), textcoords="offset points", color=col, fontsize=9)
        axs[0, j].axhline(0, color=INK, lw=0.8)
        axs[0, j].set_title("h/c %g" % hc, loc="left")
        for ax in axs[:, j]:
            ax.grid(True, color=GRID, lw=0.6)
        axs[1, j].set_xlabel("V [kn]")
    axs[0, 0].set_ylabel("g = σ/ω  [−]")
    axs[1, 0].set_ylabel("f [Hz]")
    axs[0, 0].set_xlim(*xlim); axs[0, 0].set_ylim(*glim); axs[1, 0].set_ylim(*flim)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], loc="upper right", frameon=False, markerscale=6, ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / name)
    plt.close(fig)


# E3: rev E (UVLM + BEM delta) e test P
runs = {}
for d in glob.glob(str(GW / "runs" / "GOLAND_water_c0p294_s1p029_*eifrac0p24_*delta*")):
    for hc in HC:
        if d.endswith("hc" + tag(hc)):
            runs[hc] = glob.glob(d + "/scan_*.txt")
figura("vgvf_E3.png", "E3 – tutte le radici; tratteggio = velocità di flutter del ramo di torsione (20–40 Hz)",
       {hc: load(runs.get(hc, [])) for hc in HC}, {hc: load([TP / f"scan_img_hc{tag(hc)}.txt"]) for hc in HC},
       (20, 40), (36, 52), (0, 60), (-0.15, 0.25))

# caso B: rev E (4-14 kn) + riferimento basso a h/c 0.4; con immagine 2-7.6 e 7.5-12 kn
CB = GW / "casoB_numericSteel_revE"
senza = {hc: load(glob.glob(str(CB / f"*delta*hc{tag(hc)}" / "scan_delta.txt")) +
                  ([TP / "casoB" / "scan_noimg_basse_hc0p4.txt"] if hc == 0.4 else [])) for hc in HC}
con = {hc: load([TP / "casoB" / f"scan_img_basse_hc{tag(hc)}.txt", TP / "casoB" / f"scan_img_hc{tag(hc)}.txt"]) for hc in HC}
figura("vgvf_casoB.png", "Caso B – tutte le radici; tratteggio = prima velocità di flutter fra 2.5 e 6 Hz",
       senza, con, (2.5, 6), (2, 12), (0, 10), (-0.15, 0.25))
