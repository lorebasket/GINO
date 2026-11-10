"""Replica del limite phi = 0 (solo immagine biplano, G°) contro van Walree 1999, Fig. 3.9a (10/11/2026).

Van Walree (tesi TU Delft 1999, p. 143, Fig. 3.9a): ala rettangolare AR = 6, N = 4, M = 16, tip inset 0.25;
rapporto C_La(h/c)/C_La(h/c = 4) contro Fn_h = U/sqrt(g h), per h/c = 0.25, 0.5, 1, 2. Curve: Hydvlm
(VLM con funzione di Green stazionaria completa: immagine + onde). Punti: misure di Wilson (1983).
Valori digitalizzati qui sotto dalla pagina scansionata (precisione circa +-0.01).

Il nostro modello (immagine concorde a +2h, phi = 0) non dipende da Fn_h: e' l'asintoto Fn_h -> inf.
Stessa costruzione di testO/fattore_immagine.py, di naca1232/ e dell'ala clone SHARPy.

    python replica_vanwalree.py   ->  replica_vanwalree.json, replica_vanwalree_fig3_9.png
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("fi", HERE.parents[1] / "testO" / "fattore_immagine.py")
fi = importlib.util.module_from_spec(_s)
with contextlib.redirect_stdout(io.StringIO()):
    _s.loader.exec_module(fi)


def cl_alpha(AR, hc, N, M, inset=0.25, far=1e4):
    """C_L/alpha [1/rad], corda 1, ferri di cavallo con scia piana; inset = rientro alle estremita' in
    frazioni di pannello (van Walree: 0.25). hc = inf: fluido infinito."""
    b = float(AR)
    ys = np.linspace(-b / 2, b / 2, M + 1)
    d = inset * (ys[1] - ys[0])
    ys = np.linspace(-b / 2 + d, b / 2 - d, M + 1)
    xs = np.linspace(0, 1.0, N + 1)
    pan = [(i, j) for i in range(N) for j in range(M)]
    cps = np.array([[xs[i] + 0.75 * (xs[i + 1] - xs[i]), 0.5 * (ys[j] + ys[j + 1]), 0.0] for i, j in pan])
    dy = np.array([ys[j + 1] - ys[j] for i, j in pan])

    def infl(z):
        A = np.zeros((len(pan), len(pan)))
        for m, (i, j) in enumerate(pan):
            xq = xs[i] + 0.25 * (xs[i + 1] - xs[i])
            a, bb = np.array([xq, ys[j], z]), np.array([xq, ys[j + 1], z])
            v = fi.seg(cps, a + [far, 0, 0], a) + fi.seg(cps, a, bb) + fi.seg(cps, bb, bb + [far, 0, 0])
            A[:, m] = v[:, 2]
        return A
    A = infl(0.0)
    if np.isfinite(hc):
        A = A + infl(2 * hc)   # immagine concorde a +2h
    G = np.linalg.solve(A, -np.ones(len(pan)))
    return 2 * np.sum(G * dy) / b


# --- digitalizzazione di Fig. 3.9a (van Walree 1999, p. 143) -------------------------------------------
VW_CURVE = {  # Hydvlm: (Fn_h, rapporto)
    2.0: [(1.5, 0.915), (2, 0.918), (3, 0.955), (4, 0.962), (6, 0.966), (8, 0.970), (10, 0.973)],
    1.0: [(1.5, 0.803), (2, 0.814), (3, 0.860), (4, 0.882), (6, 0.899), (8, 0.906), (10, 0.908)],
    0.5: [(1.5, 0.603), (2, 0.605), (3, 0.685), (4, 0.734), (6, 0.777), (8, 0.792), (10, 0.797)],
    0.25: [(1.5, 0.497), (2, 0.400), (3, 0.465), (4, 0.540), (6, 0.610), (8, 0.640), (10, 0.651)],
}
WILSON = {  # misure di Wilson (1983): (Fn_h, rapporto)
    2.0: [(1.5, 0.913), (2, 0.913), (2.5, 0.940), (3, 0.972)],
    1.0: [(1.0, 0.887), (2, 0.816), (3, 0.843), (4, 0.868)],
    0.5: [(2, 0.610), (3, 0.667), (4, 0.709), (5, 0.755), (6, 0.780)],
    0.25: [(3, 0.447), (4, 0.521), (5, 0.563), (6, 0.591), (7, 0.624), (8, 0.637)],
}


def main():
    N, M = 4, 16
    c4 = cl_alpha(6, 4.0, N, M)
    phi0 = {hc: cl_alpha(6, hc, N, M) / c4 for hc in VW_CURVE}
    conv = {hc: cl_alpha(6, hc, 8, 48) / cl_alpha(6, 4.0, 8, 48) for hc in VW_CURVE}
    res = {}
    for hc, pts in VW_CURVE.items():
        (f6, r6), (f10, r10) = [p for p in pts if p[0] == 6][0], pts[-1]
        a = (r10 - r6) / (1 / f6 ** 2 - 1 / f10 ** 2)            # r = r_inf - a/Fn^2 sugli ultimi punti
        r_inf = r10 + a / f10 ** 2
        wl = WILSON[hc]
        res[hc] = dict(phi0_N4M16=phi0[hc], phi0_N8M48=conv[hc], vw_Fn10=r10, vw_estrapolato=r_inf, a_1suFn2=a,
                       wilson=wl, scarto_phi0_vs_vw_inf=phi0[hc] - r_inf)
        print(f"h/c {hc:4}: phi=0 {phi0[hc]:.3f} (N8M48 {conv[hc]:.3f}) | van Walree Fn_h 10 {r10:.3f}, "
              f"estrapolato Fn_h->inf {r_inf:.3f} | scarto {phi0[hc]-r_inf:+.3f}")
    json.dump(dict(caso="AR 6, N 4, M 16, inset 0.25; rapporto a h/c=4", risultati={str(k): v for k, v in res.items()},
                   cla_h4_su_inf=c4 / cl_alpha(6, np.inf, N, M)), open(HERE / "replica_vanwalree.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    col = {2.0: "#2E6FB7", 1.0: "#1F9E89", 0.5: "#E0622B", 0.25: "#8E6BB8"}
    ax.axvspan(1.7, 2.3, color="#E0622B", alpha=0.10, lw=0)
    ax.axvspan(9, 12, color="#1F9E89", alpha=0.10, lw=0)
    ax.text(2.0, 1.035, "caso B\nal flutter", ha="center", va="top", fontsize=9, color="#9a3d14")
    ax.text(10.5, 1.035, "E3 al flutter\n(Fn_h 9–22)", ha="center", va="top", fontsize=9, color="#136b5c")
    for hc in VW_CURVE:
        x, y = zip(*VW_CURVE[hc]); ax.plot(x, y, "-", color=col[hc], lw=1.8, label="van Walree 1999, metodo completo (immagine + onde)" if hc == 2.0 else None)
        x, y = zip(*WILSON[hc]); ax.plot(x, y, "o", color=col[hc], mec="k", ms=6, label="misure di Wilson (1983)" if hc == 2.0 else None)
        ax.plot([10.4, 12], [phi0[hc]] * 2, "--", color=col[hc], lw=2.2, label="nostra immagine φ = 0 (Fn_h → ∞)" if hc == 2.0 else None)
        ax.text(12.1, phi0[hc], f"h/c {hc:g}", va="center", fontsize=9, color=col[hc])
    ax.set_xlim(0, 13.2); ax.set_ylim(0.38, 1.05)
    ax.set_xlabel("Fn_h = U/√(g h)"); ax.set_ylabel("C_Lα(h/c) / C_Lα(h/c = 4)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("AR 6 rettangolare: van Walree 1999 Fig. 3.9a, misure di Wilson e limite φ = 0", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(HERE / "replica_vanwalree_fig3_9.png", dpi=200)


if __name__ == "__main__":
    main()
