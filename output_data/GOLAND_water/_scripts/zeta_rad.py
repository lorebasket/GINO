"""Test F (T0.3), definizione rev A (14/10/2026): zeta_rad,i = B_ii / [2 (M_ii + A_ii) w]
valutata al punto di flutter (w_f del ramo 2), M_ii = 1 (modi normalizzati in massa, M_hat = I).
Lo script della rev precedente non e' stato conservato: i valori non sono confrontabili.
    python zeta_rad.py <cartella_run>...   -> max sui modi"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
src = open(os.path.join(os.path.dirname(__file__), "plot_due_casi.py")).read().split("CASI={")[0]
exec(src)
def zr(d):
    z = np.load(d + "/capytaine/modal_radiation_AB.npz", allow_pickle=True)
    w, A, B = z["omega"], z["added_mass"], z["added_damping"]
    V, g, f, m = load(d); vf, ff = cross(V, g, f, m)
    if vf is None: return None
    om = 2 * np.pi * ff
    return max(np.interp(om, w, B[:, i, i]) / (2 * (1 + np.interp(om, w, A[:, i, i])) * om) for i in range(2))   # i 2 modi del p-k
for d in sys.argv[1:]:
    if "UVLMpuro" in d: continue
    z = zr(d); print(d.split("_hc")[-1], "-" if z is None else "%.2e" % z)
