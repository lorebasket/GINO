"""Test P: confronto delle tabelle GAF con ala immagine rispetto alla tabella in fluido infinito.

    python confronta_gaf.py <riferimento.npz> <tabella.npz> [<tabella.npz> ...]

Per ogni tabella stampa lo scarto relativo su tutto il campo di k, il rapporto Q/Q_rif a k = 0
(quasi stazionario: da confrontare con il fattore immagine del test O) e la massa non circolatoria
A_nc ricavata come nel metodo delta (Re Q = a + (2/rho) A k^2 per k in [14, 24] 1/m).
"""
import sys

import numpy as np

RHO = 997.0


def anc(d, lo=14.0, hi=24.0):
    k, Q = d["k_list"], d["Q"]
    s = (k >= lo) & (k <= hi)
    X = np.stack([np.ones(s.sum()), k[s] ** 2], 1)
    nm = Q.shape[1]
    A = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            A[i, j] = np.linalg.lstsq(X, Q[s, i, j].real, rcond=None)[0][1] * RHO / 2
    return A


ref = np.load(sys.argv[1])
A0 = anc(ref)
print("riferimento", sys.argv[1], "A_nc diag", np.round(np.diag(A0), 4))
for fn in sys.argv[2:]:
    d = np.load(fn)
    assert np.allclose(d["k_list"], ref["k_list"]), "griglie k diverse"
    dQ = d["Q"] - ref["Q"]
    rel = np.linalg.norm(dQ, axis=(1, 2)) / np.linalg.norm(ref["Q"], axis=(1, 2))
    i1 = int(np.argmin(np.abs(ref["k_list"] - float(__import__("os").environ.get("KF", 8.3)))))  # k*b ~ 1.2: flutter di E3 (29 Hz a 22 m/s, k = omega/V)
    print("\n==", fn, "h =", float(d.get("image_depth", 0.0)), "m")
    print("  |dQ|/|Q|: max %.4f, a k=0 %.4f, a k=%.2f %.4f" % (rel.max(), rel[0], ref["k_list"][i1], rel[i1]))
    print("  Q/Q_rif a k=0 (diag):", np.round(np.diag(d["Q"][0].real) / np.diag(ref["Q"][0].real), 4))
    print("  Q/Q_rif a k=%.2f (diag, modulo):" % ref["k_list"][i1],
          np.round(np.abs(np.diag(d["Q"][i1])) / np.abs(np.diag(ref["Q"][i1])), 4))
    A = anc(d)
    print("  A_nc diag", np.round(np.diag(A), 4), " dA_nc/A_nc", np.round(np.diag(A - A0) / np.diag(A0), 4))
