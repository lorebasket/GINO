"""Figure del report GOLAND_water rev E (04/11/2026): mappa S(h/c, k_w h), curve sigma(V) dalle scansioni auto-consistenti.

    python figure_revE.py            # scrive le PNG accanto a questo file
"""
import glob
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"
CASOB = HERE.parents[1] / "casoB_numericSteel_revE"
SCALE = {1: "c0p294_s1p029", 4: "c1p176", 16: "c4p704", 64: "c18p816", 128: "c37p632", 256: "c75p264", 512: "c150p528"}
HC = [2.0, 1.0, 0.5, 0.4]
COL = {2.0: "#2E6FB7", 1.0: "#E0622B", 0.5: "#1F9E89", 0.4: "#8E6BB8"}
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E3E3E3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def envelope(scan, fband=None):
    """sigma massima (e sua frequenza) per ogni V di un file scan_*.txt; fband limita la ricerca a un ramo."""
    a = np.loadtxt(scan)
    if fband:
        a = a[(a[:, 1] > fband[0]) & (a[:, 1] < fband[1])]
    V = np.unique(a[:, 0])
    s, f = [], []
    for v in V:
        r = a[a[:, 0] == v]
        i = np.argmax(r[:, 2])
        s.append(r[i, 2]); f.append(r[i, 1])
    return V, np.array(s), np.array(f)


def crossing(V, s, f):
    j = next(i for i in range(1, len(V)) if s[i - 1] < 0 <= s[i])
    vf = V[j - 1] - s[j - 1] * (V[j] - V[j - 1]) / (s[j] - s[j - 1])
    return vf, f[j - 1] + (f[j] - f[j - 1]) * (vf - V[j - 1]) / (V[j] - V[j - 1])


def runs_of(scale):
    out = {}
    for d in glob.glob(str(RUNS / f"GOLAND_water_{SCALE[scale]}_*eifrac0p24_*")):
        sc = glob.glob(d + "/scan_*.txt")
        if not sc:
            continue
        hc = float(re.search(r"hc(\d+p\d+)", d).group(1).replace("p", "."))
        out["uvlm" if "UVLMpuro" in d else hc] = sc[0]
    return out


def table():
    rows = []
    for s in SCALE:
        r = runs_of(s)
        vu, _ = crossing(*envelope(r["uvlm"]))
        for hc in HC:
            vf, ff = crossing(*envelope(r[hc]))
            kwh = (2 * math.pi * ff) ** 2 * hc * 0.294 * s / 9.81
            rows.append((s, hc, vf, ff, kwh, 100 * (vf / vu - 1), vu))
    return rows


def fig_map(rows):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    # famiglia E3: f1 > 7 Hz solo per s <= 1, cioe' k_w h >= 396 a h/c 0.4
    ax.axvspan(396, 5e3, color="#EEF5EE", zorder=0)
    ax.text(4.5e3, 27.5, "famiglia E3 con\nf1 > 7 Hz (s ≤ 1)", ha="right", va="top", color="#3C7A3C", fontsize=10)
    ax.axhline(0, color=MUTED, lw=1)
    for hc in HC:
        pts = sorted([(r[4], r[5], r[0]) for r in rows if r[1] == hc])
        x, y, s = zip(*pts)
        ax.plot(x, y, "-o", color=COL[hc], lw=2, ms=7, mec="white", mew=1.5, label=f"h/c = {hc:g}", zorder=3)
        ax.annotate(f"h/c {hc:g}", (x[0], y[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=INK, fontsize=10)
    for r in rows:
        if r[1] == 0.4:
            ax.annotate(f"s={r[0]}", (r[4], r[5]), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 5e3)
    ax.set_xlabel("k_w·h = ω_f²·h / g  (alla frequenza di flutter)")
    ax.set_ylabel("S = V_f(UVLM+BEM) / V_f(UVLM) − 1  [%]")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout()
    fig.savefig(HERE / "mappa_S_kwh.png")
    plt.close(fig)


def fig_locus(scale, fband, xlim, name, title):
    r = runs_of(scale)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.axhline(0, color=MUTED, lw=1)
    V, s, f = envelope(r["uvlm"], fband)
    ax.plot(V, s / (2 * np.pi * f), "--", color=INK, lw=2, label="UVLM-only")
    for hc in HC:
        V, s, f = envelope(r[hc], fband)
        ax.plot(V, s / (2 * np.pi * f), color=COL[hc], lw=2, label=f"UVLM+BEM h/c {hc:g}")
        vf, _ = crossing(V, s, f)
        ax.plot([vf], [0], "o", color=COL[hc], ms=8, mec="white", mew=1.5, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_xlabel("V [kn]")
    ax.set_ylabel("σ / ω del ramo di flutter  [−]")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / name)
    plt.close(fig)


def fig_casoB():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.axhline(0, color=MUTED, lw=1)
    sc = glob.glob(str(CASOB / "*UVLMpuro*hc2p0" / "scan_uvlm.txt"))
    if not sc:
        return
    V, s, f = envelope(sc[0], (2.5, 6))
    ax.plot(V, s / (2 * np.pi * f), "--", color=INK, lw=2, label="UVLM-only")
    for hc in HC:
        d = glob.glob(str(CASOB / f"*delta*hc{str(hc).replace('.', 'p')}" / "scan_delta.txt"))
        if not d:
            continue
        V, s, f = envelope(d[0], (2.5, 6))
        ax.plot(V, s / (2 * np.pi * f), color=COL[hc], lw=2, label=f"UVLM+BEM h/c {hc:g}")
        vf, _ = crossing(V, s, f)
        ax.plot([vf], [0], "o", color=COL[hc], ms=8, mec="white", mew=1.5, zorder=4)
    ax.set_xlim(7.0, 9.5)
    ax.set_xlabel("V [kn]")
    ax.set_ylabel("σ / ω del ramo di flutter  [−]")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "casoB_sigma.png")
    plt.close(fig)


def casoB_table():
    out = {}
    sc = glob.glob(str(CASOB / "*UVLMpuro*hc2p0" / "scan_uvlm.txt"))
    if not sc:
        return out
    out["uvlm"] = crossing(*envelope(sc[0], (2.5, 6)))
    for hc in HC:
        d = glob.glob(str(CASOB / f"*delta*hc{str(hc).replace('.', 'p')}" / "scan_delta.txt"))
        if d:
            out[hc] = crossing(*envelope(d[0], (2.5, 6)))
    return out


if __name__ == "__main__":
    import json
    rows = table()
    for r in rows:
        print("s=%-4d h/c=%.1f V_f=%.2f kn f=%.4f Hz k_w*h=%.2f S=%+.1f %%" % r[:6])
    B = casoB_table()
    for k, v in B.items():
        print("caso B", k, "V_f=%.3f kn f=%.3f Hz" % v)
    json.dump({"mappa": [dict(zip(("s", "hc", "vf", "ff", "kwh", "S", "vu"), r)) for r in rows],
               "casoB": {str(k): {"vf": v[0], "ff": v[1]} for k, v in B.items()}},
              open(HERE / "dati_revE.json", "w"), indent=1)
    fig_map(rows)
    fig_locus(1, (20, 40), (36, 48), "E3_sigma.png", "E3 (s = 1, corda 0.294 m): k_w·h ≥ 400")
    fig_locus(256, (0.07, 0.16), (36, 60), "s256_sigma.png", "s = 256 (corda 75 m): k_w·h 1.4–7")
    fig_casoB()
