"""Riproduzione di NACA TR 1232 (Wadlin, Shuford, McGehee 1955), figure 14-15: rapporto delle pendenze della retta
di portanza a2/a1 in funzione della sommergenza f/c, superficie libera a velocita' supercritica (pressione costante,
phi = 0), sola superficie (niente fondo e pareti della vasca).

Modelli confrontati:
  1. NACA eq. (6)-(7): un ferro di cavallo al quarto di corda, punto di controllo ai 3/4 al centro, immagine a +2f
     con la stessa rotazione; a2/a1 = Gamma2/Gamma1 = 1 / (1 + W6/W5).
  2. 2D NACA: a2/a1 = (16 F^2 + 1) / (16 F^2 + 2), F = f/c.
  3. VLM con immagine IMPOSTA: Gamma dell'immagine = Gamma dell'ala per costruzione (A_tot = A_ala + A_imm).
  4. VLM con ala CLONE RISOLTA: copia geometrica dell'ala a +2f con incognite proprie e stessa condizione di tangenza
     (stessa incidenza): e' quello che fa image_surface.py in SHARPy. Deve dare Gamma_clone = Gamma_ala.
  5. VLM ridotto a un solo ferro di cavallo (nx = 1, un pannello in apertura): deve ridare il modello 1.
Punti sperimentali letti a occhio dalla figura 15 (+-0.01), media sulle velocita'.

    python riproduzione_naca1232.py      # scrive naca1232_fig15.png e naca1232.json accanto a questo file
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("fi", HERE.parents[1] / "testO" / "fattore_immagine.py")
fi = importlib.util.module_from_spec(_s); _s.loader.exec_module(fi)
seg = fi.seg


def naca_eq(A, F):
    """eq. (6)-(7) del rapporto, corda unitaria, s = semi-apertura"""
    c, s, f = 1.0, A / 2.0, F
    W5 = 1 / (2 * np.pi) * (2 / (s * c) * np.sqrt((c / 2) ** 2 + s ** 2) + 1 / s)
    r = np.sqrt((c / 2) ** 2 + (2 * f) ** 2 + s ** 2)
    W6 = 1 / (4 * np.pi) * (c / ((c / 2) ** 2 + (2 * f) ** 2) * s / r + 2 * s / ((2 * f) ** 2 + s ** 2) * (1 + c / (2 * r)))
    return 1 / (1 + W6 / W5)


def naca_2d(F):
    return (16 * F ** 2 + 1) / (16 * F ** 2 + 2)


class Ala:
    """ala rettangolare piatta, corda 1, reticolo nx x (2 ny); scia piana all'infinito"""

    def __init__(self, A, nx, ny, far=1e4):
        s = A / 2.0
        ys = np.linspace(-s, s, 2 * ny + 1)
        xs = np.linspace(0, 1.0, nx + 1)
        self.pan = [(i, j) for i in range(nx) for j in range(2 * ny)]
        self.hs = [(np.array([xs[i] + 0.25 * (xs[i + 1] - xs[i]), ys[j], 0.0]),
                    np.array([xs[i] + 0.25 * (xs[i + 1] - xs[i]), ys[j + 1], 0.0])) for i, j in self.pan]
        self.cp = np.array([[xs[i] + 0.75 * (xs[i + 1] - xs[i]), 0.5 * (ys[j] + ys[j + 1]), 0.0] for i, j in self.pan])
        self.dy = np.array([ys[j + 1] - ys[j] for i, j in self.pan])
        self.far = far

    def infl(self, zsrc, zcp):
        """w ai punti di controllo (quota zcp) dovuta ai ferri di cavallo unitari alla quota zsrc"""
        cp = self.cp + np.array([0, 0, zcp])
        M = np.zeros((len(cp), len(self.hs)))
        for m, (a, b) in enumerate(self.hs):
            a, b = a + np.array([0, 0, zsrc]), b + np.array([0, 0, zsrc])
            fa, fb = a + np.array([self.far, 0, 0]), b + np.array([self.far, 0, 0])
            M[:, m] = (seg(cp, fa, a) + seg(cp, a, b) + seg(cp, b, fb))[:, 2]
        return M

    def lift_inf(self):
        return np.sum(np.linalg.solve(self.infl(0, 0), -np.ones(len(self.pan))) * self.dy)

    def lift_img(self, F):
        """immagine imposta: stessa Gamma"""
        G = np.linalg.solve(self.infl(0, 0) + self.infl(2 * F, 0), -np.ones(len(self.pan)))
        return np.sum(G * self.dy)

    def lift_clone(self, F):
        """clone risolto: incognite proprie, stessa tangenza; restituisce portanza dell'ala e max |Gc/Ga - 1|"""
        n = len(self.pan)
        Arr, Arc = self.infl(0, 0), self.infl(2 * F, 0)
        Acr, Acc = self.infl(0, 2 * F), self.infl(2 * F, 2 * F)
        G = np.linalg.solve(np.block([[Arr, Arc], [Acr, Acc]]), -np.ones(2 * n))
        return np.sum(G[:n] * self.dy), np.max(np.abs(G[n:] / G[:n] - 1))


# punti sperimentali (fig. 15), A: [(f/c, a2/a1, fonte)]
ESP = {10: [(3.84, 0.965, "vasca 1"), (0.84, 0.826, "vasca 1"), (3.84, 0.99, "vasca 2"), (0.84, 0.83, "vasca 2"),
            (3.09, 0.99, "rif. 8"), (2.34, 0.975, "rif. 8"), (1.59, 0.935, "rif. 8"), (0.84, 0.85, "rif. 8"), (0.53, 0.81, "rif. 8")],
       6: [(5.00, 0.995, "rif. 9"), (1.00, 0.845, "rif. 9")],
       4: [(4.09, 0.99, "vasca 2"), (3.09, 0.985, "vasca 2"), (2.09, 0.975, "vasca 2"), (1.09, 0.91, "vasca 2"), (0.59, 0.795, "vasca 2")]}

if __name__ == "__main__":
    FF = np.array([0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    out = {}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), dpi=170, sharey=True)
    for ax, A in zip(axs, [10, 6, 4]):
        fine = Ala(A, nx=8, ny=int(4 * A))      # pannelli ~ 1/8 di corda in apertura
        one1 = Ala(A, nx=1, ny=1); one1.pan = [(0, 0)]
        s = A / 2  # un solo ferro di cavallo su tutta l'apertura, come nel rapporto
        one1.hs = [(np.array([0.25, -s, 0.0]), np.array([0.25, s, 0.0]))]
        one1.cp = np.array([[0.75, 0.0, 0.0]]); one1.dy = np.array([2 * s])
        L0, L1 = fine.lift_inf(), one1.lift_inf()
        rows = []
        for F in FF:
            li = fine.lift_img(F) / L0
            lc, dg = fine.lift_clone(F)
            rows.append(dict(F=float(F), naca=float(naca_eq(A, F)), vlm1=float(one1.lift_img(F) / L1),
                             img=float(li), clone=float(lc / L0), dG=float(dg), d2=float(naca_2d(F))))
            r = rows[-1]
            print("A %2d f/c %4.2f | NACA eq %.4f  VLM 1 ferro %.4f | VLM immagine %.4f  VLM clone %.4f (max|Gc/Ga-1| %.1e) | 2D %.4f"
                  % (A, F, r["naca"], r["vlm1"], r["img"], r["clone"], r["dG"], r["d2"]))
        out[A] = rows
        ax.plot(FF, [r["naca"] for r in rows], "-", color="#2E6FB7", lw=2, label="NACA eq. (6)-(7), 1 ferro di cavallo")
        ax.plot(FF, [r["img"] for r in rows], "-", color="#E0622B", lw=2, label="VLM, immagine imposta")
        ax.plot(FF, [r["clone"] for r in rows], "o", color="#E0622B", mfc="white", ms=6, label="VLM, ala clone risolta (SHARPy)")
        ax.plot(FF, [r["d2"] for r in rows], ":", color="#6B6B6B", lw=1.5, label="2D (NACA)")
        for k, (src, mk) in enumerate([("vasca 1", "s"), ("vasca 2", "D"), ("rif. 8", "^"), ("rif. 9", "v")]):
            p = [(f, v) for f, v, q in ESP[A] if q == src]
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
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(HERE / "naca1232_fig15.png")
    json.dump({str(k): v for k, v in out.items()}, open(HERE / "naca1232.json", "w"), indent=1)
