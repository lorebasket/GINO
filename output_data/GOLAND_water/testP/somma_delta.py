"""Test P: tabella GAF fine con l'effetto immagine calcolato su griglia ridotta.

    python somma_delta.py <fine.npz> <ridotta_senza.npz> <ridotta_con_immagine.npz> <uscita.npz>

Q(h) = Q_fine + [Q_rid,img(h) - Q_rid]. La griglia ridotta (M16) ha k di Nyquist piu' basso: oltre l'ultimo
k comune la differenza si prolunga con le forme asintotiche, Re dQ = a + b k^2 e Im dQ = c k (adattate
sugli ultimi punti con k >= 14), cosi' la finestra di fit di A_nc del metodo delta (14-24 1/m) vede
anche la massa non circolatoria dell'immagine.
"""
import sys

import numpy as np

fine, rs, ri = (np.load(f) for f in sys.argv[1:4])
assert np.allclose(rs["k_list"], ri["k_list"])
kr, dQ = rs["k_list"], ri["Q"] - rs["Q"]
kf = fine["k_list"]
nm = dQ.shape[1]
s = kr >= 14.0
out = np.zeros((len(kf), nm, nm), complex)
for i in range(nm):
    for j in range(nm):
        a, b = np.linalg.lstsq(np.stack([np.ones(s.sum()), kr[s] ** 2], 1), dQ[s, i, j].real, rcond=None)[0]
        c = np.linalg.lstsq(kr[s, None], dQ[s, i, j].imag, rcond=None)[0][0]
        re = np.where(kf <= kr[-1], np.interp(kf, kr, dQ[:, i, j].real), a + b * kf ** 2)
        im = np.where(kf <= kr[-1], np.interp(kf, kr, dQ[:, i, j].imag), c * kf)
        out[:, i, j] = re + 1j * im
save = {k: fine[k] for k in fine.files}
save["Q"] = fine["Q"] + out
save["image_depth"] = float(ri["image_depth"])
save["image_delta_grid"] = str(ri["grid"])
np.savez(sys.argv[4], **save)
print("k comune fino a %.2f 1/m (fine fino a %.1f); |dQ|/|Q_fine| max %.4f" % (
    kr[-1], kf[-1], np.max(np.linalg.norm(out, axis=(1, 2)) / np.linalg.norm(fine["Q"], axis=(1, 2)))))
