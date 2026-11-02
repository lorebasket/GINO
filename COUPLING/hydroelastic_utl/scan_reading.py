"""Lettura dell'inizio del flutter dalle scansioni auto-consistenti (file scan.txt: V[kn] f[Hz] sigma).

Per ogni velocita' si prende la radice meno smorzata (g = sigma/omega massimo) nella banda [f0, f1];
l'attraversamento g = 0 si interpola fra le velocita' DISPONIBILI piu' vicine (i buchi della
scansione, velocita' in cui il ramo non e' stato trovato, si saltano invece di interrompere la
lettura: con la vecchia lettura un buco faceva perdere il crossing). Si richiede che il ramo resti
instabile per `persist` velocita' disponibili consecutive (scarta radici isolate).
"""
import numpy as np


def least_damped(path, f0, f1, gmin=-1.0, vmin=0.0):
    a = np.loadtxt(path)
    g = a[:, 2] / (2 * np.pi * a[:, 1])
    sel = (a[:, 1] > f0) & (a[:, 1] < f1) & (g > gmin) & (a[:, 0] >= vmin)
    Vs = np.unique(a[:, 0]); gm = np.full(len(Vs), np.nan); fm = gm.copy()
    for i, v in enumerate(Vs):
        m = sel & (a[:, 0] == v)
        if m.any():
            j = np.argmax(g[m]); gm[i] = g[m][j]; fm[i] = a[m, 1][j]
    return Vs, gm, fm


def flutter_onset(path, f0, f1, gmin=-1.0, vmin=0.0, persist=2):
    """-> (V_f [kn], f_f [Hz]) oppure (nan, nan)."""
    Vs, gm, fm = least_damped(path, f0, f1, gmin, vmin)
    ok = np.isfinite(gm); V, g, f = Vs[ok], gm[ok], fm[ok]
    for i in range(len(V) - 1):
        if g[i] < 0 <= g[i + 1] and np.all(g[i + 1:i + 1 + persist] >= 0):
            t = -g[i] / (g[i + 1] - g[i])
            return V[i] + t * (V[i + 1] - V[i]), f[i] + t * (f[i + 1] - f[i])
    return np.nan, np.nan
