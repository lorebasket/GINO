"""GAF Q(k) dell'UVLM lineare di SHARPy e griglia delle frequenze ridotte."""
import numpy as np
import sharpy.linear.src.libss as libss


def k_grid(k_max, n_k):
    if n_k == 60:
        # griglia leggera (LV): 20 punti in 0.02-2, 20 in 2.2-8 (zona del flutter, k ~ 4-6), 19 fino a k_max
        return np.unique(np.concatenate([[0.0], np.linspace(0.02, 2.0, 20),
                                         np.linspace(2.2, 8.0, 20),
                                         np.linspace(8.5, k_max, 19)]))
    return np.unique(np.concatenate([[0.0], np.linspace(0.02, 2.0, 34),
                                     np.linspace(2.2, 8.0, 22),
                                     np.linspace(8.5, k_max, max(n_k - 57, 2))]))


def gaf(data, V_ref, k_list, Phi=None, Phi_out=None):
    """Q(k) per ampiezza modale. Phi=None: l'UVLM e' gia' proiettato da SHARPy.
    Phi_out: proiezione in uscita se diversa da quella in ingresso (immagine del fondo scafo)."""
    ls = data.linear.linear_system
    ls.update(float(V_ref))
    ss = ls.uvlm.ss
    facts = ss.ScalingFacts if hasattr(ss, 'ScalingFacts') else ls.uvlm.sys.ScalingFacts
    F_ref = float(facts["dyn_pressure"]) * float(facts["length"]) ** 2
    t_ref = float(facts["length"]) / float(V_ref)
    dimfac = F_ref / float(facts["dyn_pressure"])
    if Phi is not None:
        nd, nm = Phi.shape
        Bin = np.zeros((ss.inputs, 2 * nm)); Bin[:nd, :nm] = Phi; Bin[nd:2 * nd, nm:] = Phi
        Cout = np.zeros((nm, ss.outputs)); Cout[:, :nd] = (Phi if Phi_out is None else Phi_out).T
        B = ss.B @ Bin; C = Cout @ ss.C; D = Cout @ ss.D @ Bin
        red = libss.StateSpace(ss.A, B, C, D, dt=ss.dt)
    else:
        red = ss; nm = ss.outputs
    w_s = k_list * V_ref * t_ref
    H = red.freqresp(w_s)
    Q = np.array([np.conj(dimfac * (H[:nm, :nm, i] + 1j * w * H[:nm, nm:2 * nm, i]))
                  for i, w in enumerate(w_s)])
    return Q, dict(F_ref=F_ref, t_ref=t_ref, dt=float(ss.dt), states=ss.states)
