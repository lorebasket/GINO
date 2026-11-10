"""Test O (T1.7): fattore di riduzione della portanza circolatoria f(h/c) = C_Lalpha(h)/C_Lalpha(inf)
sotto una superficie a pressione nulla (phi = 0, Froude infinito): immagine CONCORDE a +h.

2D: lastra piana, vortici a 1/4 di pannello, punti di controllo a 3/4 (Glauert/DVM).
3D: reticolo di vortici a ferro di cavallo, ala rettangolare piatta (pianta E3: c 0.294, semi-apertura 1.029,
    ala intera speculare in y), scia piana all'infinito; l'immagine e' la stessa ala a +h con Gamma concorde.
Linearizzato: portanza = rho U sum(Gamma) (la velocita' orizzontale indotta dall'immagine e' di 2o ordine in alpha).

    python fattore_immagine.py
"""
import json
import numpy as np


def cl2d(hc, n=200):
    c = 1.0
    xv = (np.arange(n) + 0.25) / n * c
    xc = (np.arange(n) + 0.75) / n * c
    dx = xc[:, None] - xv[None, :]
    A = -1.0 / (2 * np.pi * dx)                      # w indotta da vortice unitario sullo stesso piano
    if np.isfinite(hc):
        dz = 2 * hc * c                              # immagine a quota +2h rispetto al foil
        A += -dx / (2 * np.pi * (dx ** 2 + dz ** 2))  # stesso segno di Gamma (concorde)
    G = np.linalg.solve(A, -np.ones(n))              # w + U*alpha = 0 con U*alpha = 1
    return G.sum()


def seg(P, A, B):
    """velocita' indotta in P da un segmento vortice A->B di intensita' unitaria (Biot-Savart)"""
    r1, r2 = P - A, P - B
    cr = np.cross(r1, r2)
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cr2 = np.sum(cr * cr, axis=-1)
    r0 = B - A
    k = np.where(cr2 > 1e-12, (np.sum(r0 * r1, -1) / n1 - np.sum(r0 * r2, -1) / n2) / (4 * np.pi * np.maximum(cr2, 1e-12)), 0.0)
    return cr * k[..., None]


def cl3d(hc, c=0.294, semi=1.029, ny=40, nx=8, far=1e4):
    ys = np.linspace(-semi, semi, 2 * ny + 1)
    xs = np.linspace(0, c, nx + 1)
    pan = [(i, j) for i in range(nx) for j in range(2 * ny)]
    def hs(z):  # ferri di cavallo: (A, B, scia) per pannello alla quota z
        out = []
        for i, j in pan:
            xq = xs[i] + 0.25 * (xs[i + 1] - xs[i])
            out.append((np.array([xq, ys[j], z]), np.array([xq, ys[j + 1], z])))
        return out
    cps = np.array([[xs[i] + 0.75 * (xs[i + 1] - xs[i]), 0.5 * (ys[j] + ys[j + 1]), 0.0] for i, j in pan])
    def infl(z):
        A = np.zeros((len(pan), len(pan)))
        for m, (a, b) in enumerate(hs(z)):
            far_a, far_b = a + np.array([far, 0, 0]), b + np.array([far, 0, 0])
            v = seg(cps, far_a, a) + seg(cps, a, b) + seg(cps, b, far_b)
            A[:, m] = v[:, 2]
        return A
    A = infl(0.0)
    if np.isfinite(hc):
        A = A + infl(2 * hc * c)   # immagine concorde a +2h rispetto al foil
    G = np.linalg.solve(A, -np.ones(len(pan)))
    dy = np.repeat(np.diff(ys)[None, :], nx, 0).ravel()
    return float(np.sum(G * dy))


if __name__ == "__main__":
    HC = [2.0, 1.0, 0.75, 0.5, 0.4]
    g2, g3 = cl2d(np.inf), cl3d(np.inf)
    rows = []
    print(" h/c   f_2D    asint.-(c/h)^2/32   f_3D(E3)")
    for hc in HC:
        f2, f3 = cl2d(hc) / g2, cl3d(hc) / g3
        rows.append({"hc": hc, "f2d": f2, "f3d": f3, "asint": 1 - 1 / (32 * hc ** 2)})
        print("%4.2f  %.4f  %.4f              %.4f" % (hc, f2, 1 - 1 / (32 * hc ** 2), f3))
    # controllo di griglia 2D/3D
    print("griglia 2D n=400 a h/c 0.4: %.4f" % (cl2d(0.4, 400) / cl2d(np.inf, 400)))
    print("griglia 3D ny=60 nx=12 a h/c 0.4: %.4f" % (cl3d(0.4, ny=60, nx=12) / cl3d(np.inf, ny=60, nx=12)))
    json.dump(rows, open("fattore_immagine.json", "w"), indent=1)
