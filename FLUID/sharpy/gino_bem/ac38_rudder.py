"""Caso SHARPy del timone AC38 — superficie portante per il ROM UVLM.

SCOPO E LIMITI. Questo caso serve a costruire il ROM aerodinamico UVLM sulla
superficie portante reale del timone. Il suo modello strutturale NON riproduce i
vincoli veri del timone: SHARPy esprime solo incastro o estremo libero, mentre il
timone e' tenuto da due boccole che bloccano alcuni gradi di liberta' e ne
lasciano liberi altri (inferiore: ux, uy, uz, thz; superiore: ux, uy). Misurato
sul modello a trave validato, approssimare con un incastro costa +10.8 % sul
primo modo, +50.1 % sul secondo e +14.1 % sulla torsione: inaccettabile come
modello strutturale.

Per questo la struttura resta in GINO, con i vincoli corretti, e questo caso
serve soltanto a generare una BASE sufficientemente ricca su cui costruire il
ROM. Il ROM viene poi ri-espresso nella base di GINO con
``native_state_coupling.apply_basis_congruence`` e risolto con
``native_coupled_eigs_at_V(M_hat, K_hat)`` presi dai modi corretti. Si prendono
quindi molti modi SHARPy (25-30), non sei: la proiezione e' accurata quanto piu'
la base e' ricca, indipendentemente dal fatto che i singoli modi coincidano.

GEOMETRIA (dal B-rep del file STEP, non da una descrizione approssimata):
  pelo libero                z =     0 mm
  inizio superficie portante z = -1350 mm   sopra e' stock carenato -> BEM
  tip                        z = -2685 mm
  corda                      286.2 -> 150.0 mm
  camber                     misurata max 0.00066 c  -> superficie PIANA
  svergolamento              misurato max 0.04 deg   -> NULLO
"""
import os
import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.geo_utils as geo_utils

# --- geometria, dalla catena STEP -> ANBA4 -----------------------------------
Z_TOP = 565.0
Z_BEARING_UP = 535.0
Z_WATER = 0.0
Z_BEARING_LOW = -85.0
Z_FOIL_TOP = -1350.0
Z_TIP = -2685.0
AXIS_X = 777.0

_HERE = os.path.dirname(os.path.abspath(__file__))
_FSI = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
FOIL_NPZ = os.path.join(_FSI, '2604 - Project Flutter', 'code', 'out', 'aero',
                        'foil_surface.npz')
SEC_DIR = os.path.join(_FSI, '2604 - Project Flutter', 'code', 'out', 'sections_v2')

# --- configurazione I_B_500 (RevD) -------------------------------------------
# Cambiano due cose e sono entrambe grosse. La struttura e' quella nuova (GK del
# cassone +11-25 %, giunzione fino al doppio). E soprattutto la SUPERFICIE
# PORTANTE non e' piu' la sola lama sotto z = -1350: la tabella Geometria_idro di
# RevD la definisce da z = -2680 fino a -90, con corda da 150 a 497 mm. Sono
# 0.834 m2 contro 0.269, cioe' tre volte tanto.
PLANFORM_CSV = os.path.join(_FSI, '2604 - Project Flutter', 'code', 'out',
                            'aero_IB500', 'planform_wetted.csv')
SEC_DIR_IB500 = os.path.join(_FSI, '2604 - Project Flutter', 'code', 'out',
                             'sections_IB500_nowrap')


def rud_shift(z_mm):
    """dx [mm] verso prua di ogni sezione del timone da AC38_RUD_SHIFT (csv z_mm,dx_mm); 0 se non impostato."""
    p = os.environ.get('AC38_RUD_SHIFT')
    if not p:
        return np.zeros_like(np.asarray(z_mm, float))
    t = np.loadtxt(p, delimiter=',', comments='#', ndmin=2)
    return np.interp(-np.asarray(z_mm, float), -t[:, 0], t[:, 1])


def load_planform_csv(path=PLANFORM_CSV):
    """Planform bagnata di RevD -> stesso dizionario di load_foil()."""
    import csv
    z, c, xle, t = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            z.append(float(r['z_mm'])); c.append(float(r['chord_mm']))
            xle.append(float(r['x_LE_mm'])); t.append(float(r['t_max_mm']))
    z = np.array(z); o = np.argsort(-z)
    xle = np.array(xle) + rud_shift(z)          # caso 32: sezioni traslate (asse elastico dritto)
    return dict(z=z[o], chord=np.array(c)[o]*1e-3, x_LE=xle[o],
                thick=np.array(t)[o]*1e-3)

RHO_WATER = 997.0


def s_of_z(z):
    """ascissa di trave [mm], 0 alla boccola superiore, crescente verso il tip"""
    return Z_BEARING_UP - np.asarray(z, float)


def load_foil(path=FOIL_NPZ):
    d = np.load(path)
    o = np.argsort(-d['z'])                      # dalla cima verso il tip
    return dict(z=d['z'][o], chord=d['chord'][o]*1e-3, x_LE=d['x_LE'][o],
                thick=d['thick'][o]*1e-3)


def load_sections(secdir=None):
    """6x6 ANBA4 trasportate sull'asse di rotazione, ordine ANBA4."""
    import glob
    secdir = secdir or SEC_DIR
    A = np.eye(6)
    ax, ay = AXIS_X*1e-3, 0.0
    A[0, 5] = ay; A[1, 5] = -ax; A[2, 3] = -ay; A[2, 4] = ax
    z, K, M = [], [], []
    for f in sorted(glob.glob(os.path.join(secdir, 'sec_z*_anba.npz'))):
        d = np.load(f)
        z.append(float(d['z_mm'][0]))
        Ai = A.copy()                              # sezione traslata di dx: asse a AXIS_X - dx nel suo frame
        axs = ax - float(rud_shift(z[-1]))*1e-3
        Ai[1, 5] = -axs; Ai[2, 4] = axs
        K.append(Ai.T @ d['K'] @ Ai); M.append(Ai.T @ d['M'] @ Ai)
    z = np.array(z); o = np.argsort(-z)
    return z[o], np.array([K[i] for i in o]), np.array([M[i] for i in o])


# ANBA4 (gx, gy, eps, kx, ky, kz)  ->  SHARPy (eps, gy, gz, torsione, flex, flex)
PERM = np.array([2, 0, 1, 5, 3, 4])


class AC38Rudder(fw.FlyingWing):
    """Timone AC38: superficie verticale immersa, pannelli solo sulla pala."""

    def __init__(self, M, N, Mstar_fact, u_inf, alpha=0.0, rho=RHO_WATER,
                 route='.', case_name='ac38_rudder', edgewise_stiffening=1.0,
                 variant='IB500'):
        self.variant = str(variant)
        self.rudder_dry = False
        if self.variant == 'IB500':
            foil = load_planform_csv()
            self._secdir = SEC_DIR_IB500
            # AC38_WL [mm]: quota del pelo libero per lo sweep di sommergenza. Il
            # timone porta pannelli solo sotto; se il pelo libero arriva alla punta
            # il timone e' tutto asciutto (restano solo i pannelli dell'elevatore).
            wl = os.environ.get('AC38_WL')
            if wl is not None:
                wl = float(wl)
                if wl <= float(np.min(foil['z'])) + 1.0:
                    self.rudder_dry = True
                elif wl < float(np.max(foil['z'])):
                    o = np.argsort(foil['z'])
                    row = {k: np.interp(wl, foil['z'][o], v[o]) for k, v in foil.items()}
                    keep = foil['z'] < wl
                    foil = {k: np.concatenate([[row[k]], v[keep]]) for k, v in foil.items()}
        else:
            foil = load_foil()
            self._secdir = SEC_DIR
        # AC38_TIP_EXT [mm]: prolunga la lama oltre la punta (corda e bordo d'attacco della punta) fino al
        # piano dell'elevatore: chiude il varco di 41 mm fra punta della lama e radice dell'elevatore, che
        # nella realta' e' occupato da bulbo e raccordo (effetto piastra d'estremita' dell'elevatore).
        ext = float(os.environ.get('AC38_TIP_EXT', '0') or 0.0)
        self.z_tip = Z_TIP - ext
        if ext > 0 and self.variant == 'IB500':
            i0 = int(np.argmin(foil['z']))
            foil = {k: np.concatenate([v, [self.z_tip if k == 'z' else v[i0]]]) for k, v in foil.items()}
        # La superficie portante comincia dove comincia la tabella, non a una
        # quota scelta a mano: cosi' se la planform cambia il caso la segue.
        self.z_foil_top = float(np.max(foil['z']))
        self._foil = foil
        # corda di riferimento fissa (radice bagnata intera), cosi' passo
        # temporale e scia non cambiano con la sommergenza
        c_ref = float(np.max(load_planform_csv()['chord'])) if self.variant == 'IB500' \
            else float(np.max(foil['chord']))
        span = float((s_of_z(self.z_tip) - s_of_z(Z_TOP)) * 1e-3)      # 3.250 m
        super().__init__(
            M=M, N=N, Mstar_fact=Mstar_fact, u_inf=u_inf, alpha=alpha, rho=rho,
            b_ref=span, main_chord=c_ref, aspect_ratio=span/c_ref,
            roll=90.0, beta=0.0, yaw=0.0, sweep=0.0, n_surfaces=1,
            physical_time=2, route=route, case_name=case_name, RollNodes=True,
        )
        self.span = span
        self.c_ref = c_ref
        self.sigma = 1.0
        # Irrigidimento nel piano della corda. I modi edgewise (flessione lungo
        # la corda) sono AERODINAMICAMENTE INERTI — un profilo sottile che si
        # muove di taglio non genera portanza — ma nella base di SHARPy cadono
        # fra quelli utili e sfasano l'appaiamento con la base di GINO: i nostri
        # modi a 129.9 (edgewise) e 131.4 Hz (torsione) distano l'1.2 % e la
        # metrica di MAC non li separa, producendo una permutazione non valida.
        # Irrigidendo di 1000x quei modi salgono fuori scala e l'ordine dei modi
        # utili coincide 1:1 fra i due codici. Lecito perche' questo modello
        # serve SOLO a generare la base per il ROM, non e' il modello strutturale.
        self.edgewise_stiffening = float(edgewise_stiffening)
        self._zsec, self._K, self._Msec = load_sections(self._secdir)

    # ---------------------------------------------------------------- nodi
    def _node_z(self, frac_foil=None):
        """Quote dei nodi, graduate.

        La pala portante e' solo il 41 % dello span, quindi una distribuzione
        uniforme sprecherebbe meta' dei pannelli sullo stock carenato dove
        l'aerodinamica e' spenta. Si mette `frac_foil` dei nodi sulla pala e il
        resto sopra, con un nodo ESATTAMENTE all'inizio della pala cosi' che il
        bordo dei pannelli coincida con l'inizio della superficie portante.
        """
        n = self.N + 1
        s0 = s_of_z(Z_TOP)*1e-3
        sf = s_of_z(self.z_foil_top)*1e-3
        s1 = s_of_z(self.z_tip)*1e-3
        if frac_foil is None:
            # Si concentrano i nodi sulla parte portante, ma non tutti: sopra
            # serve comunque una discretizzazione per la struttura. La frazione
            # segue la quota di span portante, con un margine.
            share = (s1 - sf) / (s1 - s0)
            frac_foil = float(min(0.90, max(0.60, 0.85*share + 0.15)))
        n_foil = max(4, int(round(frac_foil*(n-1))))
        n_dry = (n-1) - n_foil
        s = np.concatenate([np.linspace(s0, sf, n_dry+1)[:-1],
                            np.linspace(sf, s1, n_foil+1)])
        assert len(s) == n, (len(s), n)
        return Z_BEARING_UP - s*1e3, s

    def update_fem_prop(self):
        num_node_elem = self.num_node_elem
        n_elem = self.N // 2
        n_node = self.N + 1
        self.num_elem_tot = n_elem; self.num_elem_surf = n_elem
        self.num_node_surf = n_node; self.num_node_tot = n_node

        conn_loc = np.array([0, 2, 1], dtype=int)
        conn = np.zeros((n_elem, num_node_elem), dtype=int)
        for i in range(n_elem):
            conn[i, :] = conn_loc + i * (num_node_elem - 1)

        zn, s = self._node_z()
        self._zn = zn; self._s = s
        # un nodo esattamente sulla boccola inferiore, per poterci incastrare
        # mai il nodo di inizio della superficie portante: con I_B_500 i due
        # stanno a 5 mm (-85 e -90) e agganciare quello alla boccola lo portava
        # sopra la superficie portante -> pannelli da -155 invece che da -90
        # Si sceglie il nodo piu' vicino SOPRA l'inizio della superficie portante,
        # cosi' l'ordine dei nodi resta monotono.
        d = np.abs(zn - Z_BEARING_LOW)
        if not os.environ.get('AC38_BUG_BEAR'):   # diagnostica: aggancio vecchio (errato)
            d[zn <= self.z_foil_top + 1e-6] = np.inf
        i = int(np.argmin(d))
        zn[i] = Z_BEARING_LOW; s[i] = s_of_z(Z_BEARING_LOW)*1e-3
        self._zn = zn; self._s = s
        x = np.zeros_like(s); y = s - s[0]; z = np.zeros_like(s)
        if self.RollNodes:
            sr, cr = np.sin(self.roll_rad), np.cos(self.roll_rad)
            yo = y.copy(); y = cr*yo; z = sr*yo

        # Incastro alla BOCCOLA INFERIORE, non alla cima: e' li' che il timone
        # e' sostenuto, e un incastro in cima darebbe una mensola appesa, una
        # struttura completamente diversa (6.7 Hz contro i 10.5 veri). Resta
        # un'approssimazione — la boccola lascia liberi mx e my — ma serve solo
        # a generare una BASE per il ROM, e l'aerodinamica e' tutta oltre un
        # metro piu' in basso, dove l'errore di base e' piccolo. La qualita'
        # della proiezione va comunque MISURATA, non assunta.
        bc = -np.ones((n_node,), dtype=int)
        i_bear = int(np.argmin(np.abs(zn - Z_BEARING_LOW)))
        bc[i_bear] = 1
        self._i_clamp = i_bear
        frd = np.zeros((n_elem, num_node_elem, 3))
        frd[:, :, :] = [-1.0, 0.0, 0.0]

        self.frame_of_reference_delta = frd
        self.boundary_conditions = bc
        self.conn_glob = conn; self.conn_surf = conn
        self.surface_number = np.zeros((n_elem,), dtype=int)
        self.x = x; self.y = y; self.z = z

    # ---------------------------------------------------------------- aero
    def update_aero_prop(self):
        """Superficie piana: il profilo e' simmetrico e non svergolato, misurato
        sul B-rep (camber max 0.00066 della corda, svergolamento max 0.04 deg)."""
        flat = np.column_stack(geo_utils.interpolate_naca_camber(0., 0., 0., 0., 0.))
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros((self.num_elem_tot, self.num_node_elem), dtype=int)
        self.surface_m = self.M * np.ones((self.n_surfaces,), dtype=int)
        self.twist = np.zeros((self.num_elem_tot, 3))

        zn = self._zn
        fo = self._foil
        zf, cf, xf = fo['z'], fo['chord'], fo['x_LE']
        o = np.argsort(zf)
        chord_n = np.interp(zn, zf[o], cf[o])
        xle_n = np.interp(zn, zf[o], xf[o])
        # frazione di corda dal bordo d'attacco a cui sta l'asse della trave
        ea_n = (xle_n - AXIS_X) / (chord_n*1e3)
        dry = zn > self.z_foil_top
        if self.rudder_dry:
            dry[:] = True
        chord_n[dry] = cf[o][-1]        # valore innocuo: i pannelli sono spenti
        ea_n[dry] = 0.25

        self.chord = np.zeros((self.num_elem_tot, 3))
        self.elastic_axis = np.zeros((self.num_elem_tot, 3))
        for ie in range(self.num_elem_tot):
            nd = self.conn_glob[ie]
            self.chord[ie, :] = chord_n[nd]
            self.elastic_axis[ie, :] = ea_n[nd]

        # pannelli solo sulla pala portante: sopra, la sezione e' un corpo
        # tozzo con entrambi i bordi arrotondati e il reticolo non si applica
        self.aero_node = (zn <= self.z_foil_top) & (not self.rudder_dry)

    # ------------------------------------------------------- massa/rigidezza
    def update_mass_stiff(self):
        """Una scheda per elemento, interpolata dalle 79 sezioni ANBA4 e
        permutata nell'ordine SHARPy."""
        zn = self._zn
        zc = 0.5*(zn[self.conn_glob[:, 0]] + zn[self.conn_glob[:, 1]])
        zs = self._zsec
        o = np.argsort(zs)
        ne = self.num_elem_tot
        self.stiffness = np.zeros((ne, 6, 6))
        self.mass = np.zeros((ne, 6, 6))
        for ie, zz in enumerate(zc):
            Ki = np.array([[np.interp(zz, zs[o], self._K[o, i, j]) for j in range(6)]
                           for i in range(6)])
            Mi = np.array([[np.interp(zz, zs[o], self._Msec[o, i, j]) for j in range(6)]
                           for i in range(6)])
            Ks = self.sigma * Ki[np.ix_(PERM, PERM)]
            # Indice 5 nell'ordine SHARPy = flessione nel piano della corda.
            # DISATTIVATO per default (fattore 1.0). Serviva a spingere in alto i
            # due modi flessionali in corda, per far combaciare i primi quattro
            # modi di SHARPy coi quattro aerodinamicamente attivi di GINO. Ma
            # scalare riga e colonna 5 scala anche l'accoppiamento di quella
            # flessione con la TORSIONE: con fattore 35 il modo torsionale di
            # SHARPy scendeva a 95.7 Hz contro i 131.4 di GINO, il MAC crollava
            # a 0.67 e il coefficiente di proiezione del modo torsionale usciva
            # -31.5 invece di ~+/-1, cioe' un fattore ~1000 sulle GAF torsionali.
            # La strada giusta e' allineare SEI modi e lasciar fuori i due in
            # corda nel solutore p-k (pk_modal_dof_indices), non irrigidire la
            # struttura per comodita' dell'allineamento.
            if abs(self.edgewise_stiffening - 1.0) > 1e-12:
                r = np.sqrt(self.edgewise_stiffening)
                Ks[5, :] *= r
                Ks[:, 5] *= r
            self.stiffness[ie] = Ks
            self.mass[ie] = Mi[np.ix_(PERM, PERM)]
        self.elem_stiffness = np.arange(ne, dtype=int)
        self.elem_mass = np.arange(ne, dtype=int)


GRIDS = {
    "coarse":  (6, 20, 4),
    "default": (8, 30, 6),
    "n40c12":  (12, 40, 6),
    # I_B_500: la superficie portante e' lunga 2.59 m invece di 1.33 e la corda
    # arriva a 497 mm invece di 286. Servono piu' pannelli in apertura per non
    # perdere risoluzione, e piu' in corda perche' il numero di pannelli in
    # corda e' quello che domina l'accuratezza delle GAF.
    "n48c14":  (14, 48, 6),
    "n64c16":  (16, 64, 6),
}


def default_ac38_capy_npz(variant='IB500'):
    """Base modale di GINO. Finche' il BEM non c'e', si usa `modal_basis_dry.npz`,
    che porta i modi veri (vincoli plan_00) con massa aggiunta e smorzamento
    NULLI: nessuna fisica idrodinamica iniettata, solo la base per allineare le
    GAF. Quando il BEM e' pronto, i file di radiazione modale hanno la
    precedenza."""
    from pathlib import Path
    case = "AC38_RUDDER_IB500" if variant == 'IB500' else "AC38_RUDDER"
    p = Path(_FSI) / "FLUID" / "capytaine" / case / "results_modal_radiation"
    for name in ("modal_radiation_AB_depth_0.npz", "modal_radiation_AB.npz",
                 "modal_basis_dry.npz"):
        if (p / name).exists():
            return p / name
    raise FileNotFoundError("nessuna base modale in %s" % p)
