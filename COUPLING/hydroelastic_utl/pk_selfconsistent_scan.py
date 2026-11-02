"""Ricerca delle radici p-k AUTO-CONSISTENTI, senza inseguimento di rami.

PERCHE' SERVE. Il solutore p-k insegue un ramo alla volta: a ogni velocita'
sceglie, fra le radici dello spettro, quella che continua il ramo precedente. Se
l'inseguimento sbaglia — e sbaglia quando due rami si avvicinano, o quando la
mappa k -> omega(k) non e' contrattiva — il risultato non e' un errore evidente:
e' un ramo che sparisce e un altro che ne prende il posto. Le radici rimaste
senza padrone finiscono nel file degli "untracked", dove possono comparire con
parte reale positiva senza che nessuno sappia dire se sono flutter o rumore.

Il criterio che distingue le une dalle altre e' uno solo, e non ha bisogno di
nessun inseguimento: una radice e' FISICA se la frequenza che esce dal problema
agli autovalori coincide con quella a cui si e' valutata l'aerodinamica.

    lambda = sigma + i*omega  e' radice del sistema costruito con  k = omega/V
    <=>   Im(lambda_j(omega)) - omega = 0   per qualche ramo j

Qui si cerca quello zero per scansione fitta in omega, a ogni velocita'. Le
radici che ne escono sono tutte e sole quelle che soddisfano il p-k; il loro
sigma si legge senza ambiguita'. Nessun MAC, nessuna continuita' di ramo,
nessun seme: percio' questo e' un CONTROLLO INDIPENDENTE del risultato del
solutore, non un suo raffinamento.

Il costo e' una decomposizione agli autovalori per punto della griglia in omega,
quindi la scansione si usa per verificare, non al posto del p-k, che resta il
modo per seguire i rami e dare le curve V-g e V-f.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig


def _companion(V, rho, k, Q, M, C, K, config=None):
    """Matrice compagna del sistema modale, stessa convenzione di aero_system.Ac."""
    from . import aero_system
    return aero_system.Ac(V, rho, k, Q, M, C, K, store_contributions=False, config=config)


def selfconsistent_roots(V, rho, Qg_func, M, C, K, omega_grid,
                         config=None, tol_rel=1e-3, refine=40, hydro_func=None, accept_rel=1e-2):
    """Radici p-k auto-consistenti a una velocita'.

    -> lista di (omega, sigma) ordinata per omega.

    Si valuta, per ogni omega di prova, lo spettro del sistema costruito con
    k = omega/V, e si prende il ramo con frequenza piu' vicina a omega. La
    funzione  g(omega) = Im(lambda_vicina) - omega  cambia segno su una radice;
    il cambio di segno si raffina per bisezione.

    hydro_func(omega) -> (dM, dC): termini idrodinamici dipendenti dalla
    frequenza (es. A(w) - A_UVLM_nc e B(w) di Capytaine) sommati a M e C.
    Valutarli alla omega di prova e' esatto: alla radice omega e' quella fisica.
    """
    og = np.asarray(omega_grid, float)
    g = np.empty(og.size)
    sig = np.empty(og.size)

    def probe(w):
        k = max(float(w)/max(float(V), 1e-9), 1e-9)
        Q = np.atleast_2d(Qg_func(k, float(V), None, 0, 0))
        Mw, Cw = M, C
        if hydro_func is not None:
            dM, dC = hydro_func(float(w))
            Mw, Cw = M + dM, C + dC
        A = _companion(float(V), rho, k, Q, Mw, Cw, K, config=config)
        ev = eig(A, right=False)
        im = np.imag(ev); re = np.real(ev)
        pos = im > 1e-9
        if not np.any(pos):
            return np.nan, np.nan
        j = np.argmin(np.abs(im[pos] - w))
        return float(im[pos][j]), float(re[pos][j])

    for i, w in enumerate(og):
        wi, si = probe(w)
        g[i] = (wi - w) if np.isfinite(wi) else np.nan
        sig[i] = si

    out = []
    for i in range(og.size - 1):
        a, b = g[i], g[i+1]
        if not (np.isfinite(a) and np.isfinite(b)) or a == 0.0:
            continue
        if a * b > 0:
            continue
        lo, hi, fl = og[i], og[i+1], a
        for _ in range(refine):
            mid = 0.5*(lo + hi)
            wm, sm = probe(mid)
            if not np.isfinite(wm):
                break
            fm = wm - mid
            if fl * fm <= 0:
                hi = mid
            else:
                lo, fl = mid, fm
            if (hi - lo) <= tol_rel*max(mid, 1.0):
                break
        wr = 0.5*(lo + hi)
        wv, sv = probe(wr)
        # Un cambio di segno di g puo' venire da un SALTO fra rami (il "piu' vicino" cambia),
        # non da uno zero: li' Im(lambda) resta lontana da omega. Si accetta solo una radice
        # davvero auto-consistente, |Im(lambda) - omega| <= accept_rel * omega (bug 15/10/2026:
        # radici spurie registrate con la omega del salto, sigma di un altro ramo).
        if np.isfinite(wv) and abs(wv - wr) <= accept_rel * max(wr, 1.0):
            out.append((wr, sv))
    # radici distinte: la stessa puo' essere trovata da due intervalli contigui
    out.sort(key=lambda t: t[0])
    ded = []
    for w, s in out:
        if ded and abs(w - ded[-1][0]) <= 5e-3*max(w, 1.0):
            continue
        ded.append((w, s))
    return ded


def sweep(V_list, rho, Qg_func, M, C, K, omega_min=1.0, omega_max=None,
          n_omega=900, config=None, hydro_func=None):
    """Scansione su tutte le velocita'. -> dict con le radici per velocita'.

    La griglia in omega copre, per default, da omega_min fino al doppio della
    piu' alta frequenza strutturale: sopra non ci sono radici del sistema, e
    sotto si rischia di perdere i rami molto appesantiti dalla massa aggiunta.
    """
    wn = np.sqrt(np.abs(np.diag(np.asarray(K, float)) /
                        np.maximum(np.diag(np.asarray(M, float)), 1e-30)))
    omega_max = float(omega_max or 2.0*np.max(wn))
    og = np.geomspace(max(omega_min, 1e-3), omega_max, int(n_omega))
    res = {}
    for V in np.atleast_1d(np.asarray(V_list, float)):
        res[float(V)] = selfconsistent_roots(float(V), rho, Qg_func, M, C, K,
                                             og, config=config, hydro_func=hydro_func)
    return res


def report(res, v_knots=True, sigma_tol=0.0):
    """Riassunto testuale: prima velocita' con una radice instabile, se c'e'."""
    kn = 1.94384 if v_knots else 1.0
    first = None
    lines = []
    for V in sorted(res):
        roots = res[V]
        bad = [(w, s) for w, s in roots if s > sigma_tol]
        lines.append((V, len(roots), max([s for _, s in roots], default=np.nan),
                      bad[0] if bad else None))
        if bad and first is None:
            first = (V, bad[0])
    return first, lines
