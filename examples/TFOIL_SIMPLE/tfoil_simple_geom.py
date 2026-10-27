"""T-foil semplificato sul modello di AC38: geometria, materiale, sezioni. UNICA FONTE.

Da qui leggono la trave di GINO (structural_model._build_tfoil_simple_model), la
superficie UVLM di SHARPy (FLUID/sharpy/gino_bem/tfoil_simple.py) e i gusci del BEM
(FLUID/capytaine/run_tfoil_simple_radiation.py). Cambiare un numero qui lo cambia
ovunque.

GEOMETRIA (scelte di buon senso, dichiarate)
  pelo libero        z = 0
  timone             verticale, nessuna freccia, incastro a z = -90 mm (radice
                     bagnata di AC38); corda costante 497 mm (radice AC38) fino a
                     z = -1000, poi decresce linearmente fino a 150 mm (punta AC38)
                     al giunto, z = -2680
  elevatore          due semiali dritte (niente freccia ne' diedro) di 875 mm,
                     innestate DIRETTAMENTE sulla punta del timone (nodo condiviso,
                     niente bulbo ne' braccio); corda 420 mm alla radice (AC38) ->
                     150 mm in punta, lineare
  profilo            NACA 00tt simmetrico, t/c = 12 % ovunque

SEZIONI. Sezione piena in un materiale omogeneo: EA, EI, GJ, massa e inerzie dagli
integrali del profilo NACA. Gli assi invece si IMPONGONO, come in una sezione con
longherone e zavorra: asse elastico a 0.30c, baricentro a 0.45c (sezione piena
omogenea li avrebbe entrambi a ~0.42c, e senza distanza fra i due non c'e'
accoppiamento inerziale). L'inerzia torsionale e' riferita all'asse elastico
(trasporto mu*e^2 incluso: senza, la matrice di massa non e' definita positiva).

MATERIALE (TFS_SECTION=solid): CFRP quasi-isotropo equivalente, E = 50 GPa, G = 19 GPa,
rho = 1550 kg/m3. Tutto sovrascrivibile da variabili d'ambiente TFS_*.

SEZIONE ANIMA + SCHIUMA (TFS_SECTION=core, default). Un profilo di carbonio UNIFORME
(stessa sezione su timone ed elevatore: NACA 00tt di corda CK, spessore TCK), centrato
sull'asse della trave, porta la rigidezza; il profilo esterno (corda locale) e' schiuma
e porta forma e massa. Il centro elastico coincide col baricentro dell'anima (la
schiuma ha moduli 1000 volte piu' bassi); il baricentro della sezione cade dove lo
mettono le due densita', a poppa dell'asse perche' la schiuma si estende verso poppa:
l'accoppiamento inerziale non e' imposto, viene dalla geometria. XCM non si usa.
"""
import os
import numpy as np

_e = lambda k, d: float(os.environ.get(k, d))

# ---------------------------------------------------------------- geometria [m]
Z_ROOT = -0.090          # incastro e inizio della superficie bagnata
Z_KINK = -1.000          # fine del tratto a corda costante
Z_TIP = -2.680           # giunto a T
C_ROOT = 0.497           # corda del timone da Z_ROOT a Z_KINK
C_RTIP = 0.150           # corda del timone al giunto
ELEV_SEMISPAN = 0.875
C_EROOT = 0.420
C_ETIP = 0.150
TC = _e("TFS_TC", 0.12)                  # spessore relativo
# pelo libero [m]: 0 = T tutto immerso (timone da -90 mm); negativo = il timone buca la
# superficie e solo la parte sotto WL e' bagnata (sweep di sommergenza, come AC38)
WL = _e("TFS_WL", 0.0) * 1e-3
# spostamento dell'elevatore lungo la corrente [m], positivo a poppa (negativo = verso prua).
# Con EDX != 0 l'elevatore e' collegato alla punta del timone da un elemento rigido senza massa.
EDX = _e("TFS_EDX", 0.0)

# ---------------------------------------------------------------- assi [frazione di corda]
XEA = _e("TFS_XEA", 0.30)
XCM = _e("TFS_XCM", 0.45)

# ---------------------------------------------------------------- materiale
E_MOD = _e("TFS_E", 50e9)
G_MOD = _e("TFS_G", 19e9)
RHO_S = _e("TFS_RHO", 1550.0)
# TFS_PHYS (21/10/2026, default 1): convenzione fisica come AC38. Matrice di massa sezionale con il segno fisico
# (rotazione destrorsa: una torsione positiva abbassa i punti a poppa -> M[w, th_v] = -mu*e) e rotazioni nodali
# fisiche in GINO (beam_rotations_physical). UVLM (SHARPy) e BEM usano rotazioni fisiche: con TFS_PHYS = 0 (storico)
# vedevano l'accoppiamento flesso-torsionale rovesciato (examples/TFOIL_SIMPLE/check_torsion_handedness.py).
PHYS = os.environ.get("TFS_PHYS", "1") != "0"
SCG = -1.0 if PHYS else 1.0
K_SCALE = _e("TFS_KSCALE", 1.0)          # moltiplica E e G (rigidezze) a massa fissa
SECTION = os.environ.get("TFS_SECTION", "core")
# anima in carbonio (laminato prevalentemente unidirezionale lungo lo span) e schiuma
# anima SCALATA (default): stessa forma e stessa proporzione su tutta l'apertura, corda
# FK * c locale. Anima COSTANTE (TFS_CORE=const, corda CK): dovendo stare nella punta da
# 150 mm viene minuscola (EI ~ 1e3 N m2, primo modo ~1 Hz): non e' un foil credibile.
CORE = os.environ.get("TFS_CORE", "scaled")
FK = _e("TFS_FK", 0.5)                   # corda dell'anima / corda esterna (anima scalata)
CK = _e("TFS_CK", 0.10)                  # corda dell'anima [m] (anima costante)
TCK = _e("TFS_TCK", 0.18)                # spessore relativo dell'anima (sulla sua corda)
E_C, G_C, RHO_C = _e("TFS_EC", 110e9), _e("TFS_GC", 4.5e9), _e("TFS_RHOC", 1550.0)
E_F, G_F, RHO_F = _e("TFS_EF", 0.08e9), _e("TFS_GF", 0.03e9), _e("TFS_RHOF", 100.0)
# pelle esterna di carbonio (laminato +-45, quasi isotropo nel piano) sul contorno del profilo:
# chiude la sezione (torsione alla Bredt) e porta massa sul contorno. TFS_TSKIN = 0: niente pelle
T_SKIN = _e("TFS_TSKIN", 0.0)
E_S, G_S, RHO_S_SKIN = _e("TFS_ES", 50e9), _e("TFS_GS", 19e9), _e("TFS_RHOSK", 1550.0)
KAPPA = 5.0 / 6.0
RHO_W = 997.0

L_RUDDER = Z_ROOT - Z_TIP                # 2.590 m


def tag():
    """Nome della variante: entra nei nomi di GAF, BEM e cartelle (niente collisioni)."""
    if SECTION == "core":
        k = "fk%g" % FK if CORE == "scaled" else "ck%g" % CK
        sk = "_sk%g" % (T_SKIN * 1e3) if T_SKIN > 0 else ""
        ed = "_edx%+g" % (EDX * 1e3) if EDX else ""
        # densita' di anima e pelle nel nome solo se diverse dal default (nomi storici invariati)
        rc = ("_rc%g" % RHO_C if RHO_C != 1550.0 else "") + ("_rs%g" % RHO_S_SKIN if RHO_S_SKIN != 1550.0 else "")
        return "core_%s_tck%g_Ec%g_Gc%g_rf%g%s%s_ks%g_xea%g_tc%g%s" % (k, TCK, E_C / 1e9, G_C / 1e9, RHO_F, rc, sk, K_SCALE, XEA, TC, ed) \
            + ("_ph" if PHYS else "")
    return "E%g_G%g_r%g_ks%g_xea%g_xcm%g_tc%g" % (E_MOD / 1e9, G_MOD / 1e9, RHO_S, K_SCALE, XEA, XCM, TC) + ("_ph" if PHYS else "")


def chord_rudder(depth):
    """Corda del timone alla profondita' depth = -z [m] (>0)."""
    z = -np.asarray(depth, float)
    lin = C_ROOT + (C_RTIP - C_ROOT) * (Z_KINK - z) / (Z_KINK - Z_TIP)
    return np.where(z >= Z_KINK, C_ROOT, lin)


def chord_elev(s):
    """Corda dell'elevatore a distanza s [m] dall'asse di simmetria."""
    return C_EROOT + (C_ETIP - C_EROOT) * np.abs(np.asarray(s, float)) / ELEV_SEMISPAN


def naca_half_thickness(xi, tc=TC):
    """Semispessore NACA 00tt su corda unitaria (bordo d'uscita chiuso)."""
    xi = np.asarray(xi, float)
    return 5 * tc * (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2
                     + 0.2843 * xi**3 - 0.1036 * xi**4)


_trap = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x (GINO) e 2.x
_XI = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 401)))
_T = 2 * naca_half_thickness(_XI)                    # spessore locale / c
_A1 = _trap(_T, _XI)                              # area / c^2
_XC1 = _trap(_T * _XI, _XI) / _A1                 # centroide / c
_IF1 = _trap(_T**3 / 12, _XI)                     # I attorno alla corda / c^4
_IE1 = _trap(_T * (_XI - _XC1)**2, _XI)           # I attorno al centroide, nel piano / c^4
_J1 = _trap(_T**3 / 3, _XI)                       # Saint-Venant, striscia sottile / c^4


def _props(tc):
    """Integrali del profilo NACA 00tt su corda unitaria: area, centroide, I_flap, I_edge (centroide), J."""
    t = 2 * naca_half_thickness(_XI, tc)
    a = _trap(t, _XI); xc = _trap(t * _XI, _XI) / a
    return a, xc, _trap(t**3 / 12, _XI), _trap(t * (_XI - xc)**2, _XI), _trap(t**3 / 3, _XI)


def section(c):
    """Sezione alla corda esterna c: anima + schiuma (default) oppure piena."""
    return _section_core(c) if SECTION == "core" else _section_solid(c)


def _skin(c):
    """Pelle sottile sul contorno esterno (corda c, bordo d'attacco a -XEA*c): integrali di linea.
    Ritorna area, I_flap (attorno alla corda), I_edge e momento statico attorno all'asse, J di Bredt."""
    xi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 201)))
    yt = naca_half_thickness(xi) * c
    x = (xi - XEA) * c
    P = np.vstack([np.column_stack([x, yt]), np.column_stack([x[::-1], -yt[::-1]])[1:]])
    d = np.diff(P, axis=0); ds = np.hypot(d[:, 0], d[:, 1]); m = 0.5 * (P[1:] + P[:-1])
    t = T_SKIN
    A = t * ds.sum()
    If = t * (ds * m[:, 1] ** 2).sum()
    Ie = t * (ds * m[:, 0] ** 2).sum()
    S = t * (ds * m[:, 0]).sum()
    Aenc = 2 * _trap(yt, x)
    J = 4 * Aenc ** 2 * t / ds.sum()
    return A, If, Ie, S, J


def _section_core(c):
    """Anima di carbonio (corda CK, centroide sull'asse) dentro un profilo di schiuma di
    corda c (bordo d'attacco a XEA*c davanti all'asse). Tutto riferito all'asse."""
    ak, _, ifk, iek, jk = _props(TCK)
    ao, xco, ifo, ieo, jo = _props(TC)
    ck = FK * c if CORE == "scaled" else CK
    Ak, Ifk, Iek, Jk = ak * ck**2, ifk * ck**4, iek * ck**4, jk * ck**4
    Ao, Ifo, Jo = ao * c**2, ifo * c**4, jo * c**4
    xo = (xco - XEA) * c                                  # centroide del profilo esterno rispetto all'asse
    Ieo = ieo * c**4 + Ao * xo**2                         # I nel piano, attorno all'asse
    if Ak > Ao:
        raise ValueError("anima piu' grande del profilo esterno (c = %.3f)" % c)
    ks = K_SCALE
    EA = ks * (E_C * Ak + E_F * (Ao - Ak))
    EIf = ks * (E_C * Ifk + E_F * (Ifo - Ifk))
    EIe = ks * (E_C * Iek + E_F * (Ieo - Iek))
    GJ = ks * (G_C * Jk + G_F * (Jo - Jk))
    GA = ks * KAPPA * (G_C * Ak + G_F * (Ao - Ak))
    mu = RHO_C * Ak + RHO_F * (Ao - Ak)
    S = RHO_F * Ao * xo                                   # momento statico (anima centrata: 0)
    m33 = RHO_C * Ifk + RHO_F * (Ifo - Ifk)
    m55 = RHO_C * Iek + RHO_F * (Ieo - Iek)
    if T_SKIN > 0:
        As, Ifs, Ies, Ss, Js = _skin(c)
        EA += ks * E_S * As; EIf += ks * E_S * Ifs; EIe += ks * E_S * Ies
        GJ += ks * G_S * Js; GA += ks * KAPPA * G_S * As
        mu += RHO_S_SKIN * As; S += RHO_S_SKIN * Ss
        m33 += RHO_S_SKIN * Ifs; m55 += RHO_S_SKIN * Ies
    K = np.diag([GA, EA, GA, EIf, GJ, EIe])
    e = S / mu
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = mu
    M[1, 5] = M[5, 1] = -SCG * mu * e
    M[2, 4] = M[4, 2] = +SCG * mu * e
    M[3, 3] = m33
    M[5, 5] = m55                                         # gia' attorno all'asse (trasporto incluso)
    M[4, 4] = M[3, 3] + M[5, 5]
    return K, M


def _section_solid(c):
    """K_loc, M_loc 6x6 nel frame di sezione di GINO: u = corda (verso poppa),
    v = span, w = spessore; gdl [u, v, w, th_u, th_v (torsione), th_w].
    Riferite all'asse elastico (x = XEA*c dal bordo d'attacco)."""
    E, G = K_SCALE * E_MOD, K_SCALE * G_MOD
    A = _A1 * c**2
    If, Ie, J = _IF1 * c**4, _IE1 * c**4, _J1 * c**4
    K = np.diag([KAPPA * G * A, E * A, KAPPA * G * A, E * If, G * J, E * Ie])
    mu = RHO_S * A
    e = (XCM - XEA) * c                                  # baricentro a poppa dell'asse elastico
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = mu
    # accoppiamento traslazione-rotazione da baricentro fuori asse (stessa
    # convenzione di structural_model._apply_mass_cg_coupling)
    M[1, 5] = M[5, 1] = -SCG * mu * e
    M[2, 4] = M[4, 2] = +SCG * mu * e
    M[3, 3] = RHO_S * If                                 # rotazione attorno alla corda
    M[4, 4] = RHO_S * (If + Ie) + mu * e**2              # torsione, riferita all'asse elastico
    M[5, 5] = RHO_S * Ie + mu * e**2
    return K, M


def summary():
    rows = []
    for nm, c in (("timone radice", C_ROOT), ("timone giunto", C_RTIP), ("elev radice", C_EROOT), ("elev punta", C_ETIP)):
        K, M = section(c)
        rows.append("%-14s c=%.3f  m=%.2f kg/m  x_cg=%+.1f mm (%.2fc)  EI_flap=%.3g  EI_edge=%.3g  GJ=%.3g N m2"
                    % (nm, c, M[0, 0], 1e3 * M[2, 4] / M[0, 0], XEA + M[2, 4] / M[0, 0] / c, K[3, 3], K[5, 5], K[4, 4]))
    return "\n".join(rows)


if __name__ == "__main__":
    print(tag()); print(summary())
    print("mu* (massa/acqua del cilindro sulla corda) = %.3f" % (RHO_S * _A1 / (RHO_W * np.pi / 4)))
