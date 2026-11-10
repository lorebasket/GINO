"""Confronto SHARPy con immagine (sharpy_naca1232.json) contro NACA TR 1232: equazioni (6)-(7), VLM Python
(naca1232.json) e punti sperimentali della fig. 15. Scrive sharpy_naca1232_fig15.png e la tabella a schermo."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("rn", HERE / "riproduzione_naca1232.py")
rn = importlib.util.module_from_spec(_s); _s.loader.exec_module(rn)
sh = json.load(open(HERE / "sharpy_naca1232.json"))
py = json.load(open(HERE / "naca1232.json"))


def curva(A, M):
    L0 = [r["L"] for r in sh if r["A"] == A and r["M"] == M and r["F"] == 0][0]
    pts = sorted((r["F"], r["L"] / L0) for r in sh if r["A"] == A and r["M"] == M and r["F"] > 0)
    return np.array(pts).T


print("Controllo di codice: SHARPy M8 contro VLM Python (stessa pannellatura 8 x 8A)")
for A in rn.ESP:
    F8, r8 = curva(A, 8)
    d = [abs(np.interp(p["F"], F8, r8) - p["img"]) for p in py[str(A)] if p["F"] in F8]
    print(f"  A {A:2d}: max |SHARPy - VLM| = {max(d):.4f}")
print("\nEsperimenti (fig. 15) contro modelli; scarti in punti di a2/a1")
print(f"{'A':>3} {'f/c':>5} {'fonte':>8} {'esp':>6} {'SHARPy':>7} {'M8-M16':>7} {'NACA':>6} | {'Δ SHARPy':>8} {'Δ NACA':>7}")
tot = {"s": [], "n": []}
for A, pts in rn.ESP.items():
    F16, r16 = curva(A, 16); F8, r8 = curva(A, 8)
    for f, v, src in sorted(pts):
        s = np.interp(f, F16, r16); n = rn.naca_eq(A, f)
        tot["s"].append(s - v); tot["n"].append(n - v)
        print(f"{A:3d} {f:5.2f} {src:>8} {v:6.3f} {s:7.3f} {np.interp(f, F8, r8) - s:+7.4f} {n:6.3f} | {100 * (s - v):+8.1f} {100 * (n - v):+7.1f}")
for k, lab in (("s", "SHARPy"), ("n", "NACA eq.")):
    e = 100 * np.array(tot[k])
    print(f"{lab:>9}: scarto medio {e.mean():+.1f}, rms {np.sqrt((e ** 2).mean()):.1f}, max |.| {abs(e).max():.1f} punti")

FF = np.linspace(0.2, 5.2, 200)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), dpi=170, sharey=True)
for ax, A in zip(axs, [10, 6, 4]):
    F16, r16 = curva(A, 16)
    ax.plot(FF, rn.naca_eq(A, FF), "-", color="#2E6FB7", lw=2, label="NACA eq. (6)-(7)")
    ax.plot(F16, r16, "-o", color="#E0622B", lw=2, ms=4, label="SHARPy con ala immagine")
    ax.plot(FF, rn.naca_2d(FF), ":", color="#6B6B6B", lw=1.5, label="2D (NACA)")
    for src, mk in [("vasca 1", "s"), ("vasca 2", "D"), ("rif. 8", "^"), ("rif. 9", "v")]:
        p = [(f, v) for f, v, q in rn.ESP[A] if q == src]
        if p:
            ax.plot(*zip(*p), mk, color="#222222", ms=6, label="esperimento, " + src)
    ax.set_title("A = %d" % A, loc="left"); ax.set_xlabel("sommergenza f/c")
    ax.grid(True, color="#E3E3E3", lw=0.7); ax.set_xlim(0, 5.2); ax.set_ylim(0.6, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("a2/a1  (pendenza a f / pendenza in fluido infinito)")
h, l = [], []
for ax in axs:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(HERE / "sharpy_naca1232_fig15.png")
