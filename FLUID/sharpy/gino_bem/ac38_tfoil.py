"""Caso SHARPy del timone AC38 a T (I_B_500 + elevatore) — superficie portante UVLM.

SCOPO. Serve SOLO la superficie portante: tre superfici (lama del timone e due
semiali dell'elevatore) per l'UVLM. La struttura vera del T sta in GINO
(`_build_ac38_tfoil_model`, vincoli delle boccole, aggancio rigido sull'asse di
simmetria). Le GAF si ottengono proiettando la risposta UVLM NODALE sulle forme
modali di GINO, quindi la struttura dentro SHARPy non entra nel risultato: qui
e' di comodo (le 6x6 del timone e, per l'elevatore, quella della punta della
lama), basta che il caso si carichi.

GEOMETRIA
  timone      : come `ac38_rudder.AC38Rudder` (planform bagnata di RevD), ma con
                le sezioni contatto + carenatura.
  elevatore   : planform della sola ala (mat5, decisione 5 dell'handoff) da
                `out/aero_ELEV/planform_wing.csv`, bordi d'attacco/uscita veri per
                ogni y; asse della trave sul centro di tensione di
                `out/beam_ELEV/elevator_beam.npz` (freccia e diedro inclusi).
  innesto     : elemento di collegamento rigido SENZA pannelli dalla punta della
                lama alla radice vera dell'elevatore (135.5 mm a poppa, 46 mm
                sotto, come il collegamento di GINO); le semiali partono da li',
                alla quota e alla corda vere.
"""
import os
import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

from gino_bem.ac38_rudder import (AC38Rudder, AXIS_X, Z_TOP, Z_TIP, load_sections,
                                  default_ac38_capy_npz)  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_FSI = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_CODE = os.path.join(_FSI, '2604 - Project Flutter', 'code', 'out')
SEC_DIR_FAIR = os.path.join(_CODE, 'sections_IB500_nowrap_contact_fair')
ELEV_PLANFORM = os.path.join(_CODE, 'aero_ELEV', 'planform_wing.csv')
# AC38_ELEV_BEAM: tabella alternativa (S04/S06): la stessa che legge GINO (config_AC38_TFOIL), cosi' la linea di
# trave di SHARPy segue il centro di tensione quando cambia la rigidezza dell'ala
ELEV_BEAM = os.environ.get('AC38_ELEV_BEAM') or os.path.join(_CODE, 'beam_ELEV', 'elevator_beam.npz')

GRIDS = {
    # (M corda, N timone, Mstar, N per semiala)
    "n48c14e20": (14, 48, 6, 20),
    # convergenza (22/10/2026, confronto con VLM/URANS DT: Im Q44 ad alto k)
    "n48c20e20": (20, 48, 6, 20),
    "n48c14e20w10": (14, 48, 10, 20),
    # passo di scia dimezzato (dt = c/28, 5o valore = fattore sul passo), scia 3 corde per tenere i pannelli di scia
    "n48c14e20d2": (14, 48, 3, 20, 2.0),
    "n48c14e20d3": (14, 48, 3, 20, 3.0),       # passo c/42, scia 3 corde
    "n48c14e20d4": (14, 48, 2, 20, 4.0),       # passo c/56, scia 2 corde (memoria)
    # griglia media per il confronto con/senza fondo scafo (l'immagine raddoppia i pannelli della lama)
    "n32c12e16": (12, 32, 6, 16),
    "coarse":    (8, 30, 6, 12),
}


def elev_shift(y_mm):
    """dx [mm] verso prua di ogni sezione dell'elevatore da AC38_ELEV_SHIFT (csv y_mm,dx_mm); 0 se non impostato."""
    p = os.environ.get('AC38_ELEV_SHIFT')
    if not p:
        return np.zeros_like(np.asarray(y_mm, float))
    t = np.loadtxt(p, delimiter=',', comments='#', ndmin=2)
    return np.interp(np.abs(y_mm), t[:, 0], t[:, 1])


def load_elevator():
    pf = np.loadtxt(ELEV_PLANFORM, delimiter=',', skiprows=1)
    b = np.load(ELEV_BEAM)
    # AC38_ELEV_DX [mm]: elevatore traslato verso prua (x CAD crescente) per lo stagger
    dx = float(os.environ.get('AC38_ELEV_DX', '0') or 0.0)
    yb = np.asarray(b['y_mm'], float)
    # AC38_ELEV_SHIFT=<csv y_mm,dx_mm>: in piu' ogni sezione traslata di dx(y) (asse elastico dritto)
    sh = elev_shift(pf[:, 0]); shb = elev_shift(yb)
    return dict(y=pf[:, 0], le=pf[:, 1] + dx + sh, te=pf[:, 2] + dx + sh,
                yb=yb,
                xtc=np.asarray(b['tension_center'], float)[:, 0] + dx + shb,
                ztc=np.asarray(b['tension_center'], float)[:, 1])


class AC38TFoil(AC38Rudder):
    """Timone AC38 I_B_500 + due semiali dell'elevatore (T)."""

    def __init__(self, M, N, Mstar_fact, u_inf, N_elev=20, **kw):
        if N_elev % 2:
            raise ValueError("N_elev deve essere pari (elementi a 3 nodi)")
        self.N_elev = int(N_elev)
        self._elev = load_elevator()
        kw.setdefault('case_name', 'ac38_tfoil')
        super().__init__(M, N, Mstar_fact, u_inf, variant='IB500', **kw)
        # AC38_RUD_SECDIR: sezioni del timone alternative (caso 35, spessore scalato), le stesse della trave GINO
        self._secdir = os.environ.get('AC38_RUD_SECDIR') or SEC_DIR_FAIR
        self._zsec, self._K, self._Msec = load_sections(self._secdir)

    # ------------------------------------------------------------------ params
    def update_derived_params(self):
        """Come `Model2tTfoilStrut`: il FlyingWing dimensiona le superfici prima
        che i rami esistano, quindi si costruisce prima la topologia completa."""
        self.dt = self.main_chord / self.M / self.u_inf * self.dt_factor
        self.n_tstep = int(np.round(self.physical_time / self.dt))
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.roll_rad = self.roll * np.pi / 180.0
        self.beta_rad = self.beta * np.pi / 180.0
        self.sweep_rad = self.sweep * np.pi / 180.0
        self.yaw_rad = self.yaw * np.pi / 180.0
        self.quat = algebra.euler2quat(np.array([self.roll_rad, self.alpha_rad, self.yaw_rad]))
        self.wing_span = self.aspect_ratio * self.main_chord
        self.S = self.wing_span * self.main_chord
        self.update_fem_prop()
        self.control_surface = np.zeros((self.num_elem_tot, self.num_node_elem), dtype=int) - 1
        self.control_surface_type = np.zeros((self.n_control_surfaces,), dtype=int)
        self.control_surface_deflection = np.zeros((self.n_control_surfaces,))
        self.control_surface_chord = np.array([self.M // 2], dtype=int)
        self.control_surface_hinge_coord = np.zeros_like(self.control_surface_type, dtype=float)
        self.update_mass_stiff()
        self.update_aero_prop()

    # ------------------------------------------------------------------ FEM
    def update_fem_prop(self):
        super().update_fem_prop()
        self.n_rudder_node = self.num_node_tot
        self.n_rudder_elem = self.num_elem_tot
        tip = self.num_node_tot - 1
        self.tip_idx = tip
        self.boundary_conditions[tip] = 0          # innesto del T, non piu' estremo libero
        tipx, tipy, tipz = self.x[tip], self.y[tip], self.z[tip]
        # direzione dello span del timone (dopo RollNodes) e direzione laterale
        e_span = np.array([self.x[tip] - self.x[0], self.y[tip] - self.y[0],
                           self.z[tip] - self.z[0]])
        e_span /= np.linalg.norm(e_span)
        e_lat = np.cross(e_span, [1.0, 0.0, 0.0])
        e_lat /= np.linalg.norm(e_lat)
        self._e_span, self._e_lat = e_span, e_lat

        E = self._elev
        ye = np.linspace(0.0, E['y'][-1], self.N_elev + 1)       # [mm]
        xtc = np.interp(ye, E['yb'], E['xtc'])
        ztc = np.interp(ye, E['yb'], E['ztc'])
        # nel CAD la corrente va verso -x (bordo d'attacco a x MASSIMO, verificato
        # dallo spessore dei bordi); x di SHARPy e' nel verso della corrente
        dx = -(xtc - xtc[0])*1e-3                   # freccia, origine alla radice
        dsp = -(ztc - ztc[0])*1e-3                  # diedro, verso la punta del timone (+span)
        self._ye = ye; self._xtc_e = xtc

        n_el = self.N_elev // 2
        conn_loc = np.array([0, 2, 1], dtype=int)
        X, Y, Z = [self.x], [self.y], [self.z]
        conn, bc = [self.conn_glob], [self.boundary_conditions]
        frd, surf = [self.frame_of_reference_delta], [self.surface_number]
        nn, ne = self.num_node_tot, self.num_elem_tot
        # COLLEGAMENTO RIGIDO (RevD nota 13), come in GINO: dalla punta della lama
        # alla radice VERA dell'elevatore (centro di tensione a y = 0), 135.5 mm a
        # poppa e 46 mm sotto. Elemento SENZA pannelli (surface -1): cosi' i pannelli
        # dell'elevatore stanno alla quota e alla corda vere, e non nascono pannelli
        # spuri sul collegamento.
        s0 = self.z[0]
        C = np.array([-(xtc[0] - AXIS_X)*1e-3, tipy, s0 + (Z_TOP - ztc[0])*1e-3])
        Cm = 0.5*(C + np.array([tipx, tipy, tipz]))
        iC, iCm = nn, nn + 1
        X.append([C[0], Cm[0]]); Y.append([C[1], Cm[1]]); Z.append([C[2], Cm[2]])
        conn.append(np.array([[tip, iC, iCm]])); bc.append(np.zeros(2, dtype=int))
        f = np.zeros((1, self.num_node_elem, 3)); f[:, :, :] = [-1.0, 0.0, 0.0]
        frd.append(f); surf.append(np.array([-1]))
        self.link_elem = ne; self.root_idx = iC
        nn += 2; ne += 1
        self.elev_elem_ranges, self.elev_nodes = {}, {}
        for sign, sid, lab in ((+1.0, 1, 'port'), (-1.0, 2, 'stbd')):
            g = np.empty(self.N_elev + 1, dtype=int)
            g[0] = iC; g[1:] = nn + np.arange(self.N_elev)
            P = (C[None, :]
                 + dx[1:, None]*np.array([1.0, 0, 0])[None, :]
                 + dsp[1:, None]*e_span[None, :]
                 + sign*ye[1:, None]*1e-3*e_lat[None, :])
            X.append(P[:, 0]); Y.append(P[:, 1]); Z.append(P[:, 2])
            c = np.array([g[conn_loc + 2*i] for i in range(n_el)])
            if lab == 'port' and os.environ.get('AC38_ELEV_REVERSE'):
                # prova: elementi percorsi dalla punta verso la radice, cosi' le
                # normali dei pannelli di questa semiala si ribaltano
                c = c[::-1][:, [1, 0, 2]]
            conn.append(c)
            b = np.zeros(self.N_elev, dtype=int); b[-1] = -1
            bc.append(b)
            f = np.zeros((n_el, self.num_node_elem, 3)); f[:, :, :] = [-1.0, 0.0, 0.0]
            frd.append(f); surf.append(np.full(n_el, sid, dtype=int))
            self.elev_elem_ranges[lab] = (ne, ne + n_el)
            self.elev_nodes[lab] = g
            nn += self.N_elev; ne += n_el
        self.x = np.concatenate(X); self.y = np.concatenate(Y); self.z = np.concatenate(Z)
        self.conn_glob = np.vstack(conn); self.conn_surf = self.conn_glob
        self.boundary_conditions = np.concatenate(bc)
        self.frame_of_reference_delta = np.vstack(frd)
        self.surface_number = np.concatenate(surf)
        self.num_node_tot = len(self.x); self.num_elem_tot = len(self.conn_glob)
        self.n_surfaces = 3
        self.image_map = {}
        if os.environ.get('AC38_IMAGE') and not self.rudder_dry:
            self._add_image()
        if self.rudder_dry:
            # timone tutto fuori dall'acqua: niente pannelli, elevatore = superfici 0 e 1
            sn = self.surface_number
            sn[:self.n_rudder_elem] = -1
            sn[sn > 0] -= 1
            self.n_surfaces = 2

    def _add_image(self):
        """Fondo scafo (AC38_IMAGE=wall: parete rigida, immagine con moto speculare;
        AC38_IMAGE=fs: superficie libera ad alta frequenza, p = 0, moto antispeculare).
        La lama bagnata viene specchiata rispetto al piano z = z del nodo d'inizio della
        superficie portante (-90 mm CAD). L'immagine e' la superficie 3, appesa alla cima
        del timone con un elemento SENZA pannelli: le sue forze restano sui suoi nodi, che
        la proiezione esclude in uscita (gino_to_sharpy). Immagine dell'elevatore omessa
        (a ~5 m dall'elevatore vero)."""
        nr = self.n_rudder_node
        zn = self._zn
        i_f = int(np.flatnonzero(zn <= self.z_foil_top + 1e-6)[0])
        i_e = i_f - (i_f % 2)                      # bordo d'elemento a monte (nodi 0, 2, 4, ...)
        zf = self.z[i_f]
        real = np.arange(i_e, nr)                  # nodi da specchiare (ordine lungo lo span)
        nn, ne = self.num_node_tot, self.num_elem_tot
        # collegamento [nodo 0 (cima del timone), radice dell'immagine, nodo di mezzo]
        P0 = np.array([self.x[0], self.y[0], self.z[0]])
        Pr = np.array([self.x[i_e], self.y[i_e], 2*zf - self.z[i_e]])
        iLm = nn
        Xn, Yn, Zn = [0.5*(P0[0] + Pr[0])], [0.5*(P0[1] + Pr[1])], [0.5*(P0[2] + Pr[2])]
        img = nn + 1 + np.arange(len(real)); root = img[0]
        X = [self.x, Xn, self.x[real]]; Y = [self.y, Yn, self.y[real]]; Z = [self.z, Zn, 2*zf - self.z[real]]
        conn = [self.conn_glob, np.array([[0, root, iLm]])]
        f = np.zeros((1, self.num_node_elem, 3)); f[:, :, :] = [-1.0, 0.0, 0.0]
        frd = [self.frame_of_reference_delta, f]; surf = [self.surface_number, np.array([-1])]
        bc = self.boundary_conditions.copy(); bc[0] = 0
        bc = np.concatenate([bc, [0], np.zeros(len(real), dtype=int)]); bc[-1] = -1
        # elementi dell'immagine: stessi elementi del timone fra i_e e la punta
        rmap = {int(r): int(m) for r, m in zip(real, img)}
        el_r = [ie for ie in range(self.n_rudder_elem) if all(int(n) in rmap for n in self.conn_glob[ie])]
        ci = np.array([[rmap[int(n)] for n in self.conn_glob[ie]] for ie in el_r])
        conn.append(ci)
        f = np.zeros((len(el_r), self.num_node_elem, 3)); f[:, :, :] = [-1.0, 0.0, 0.0]
        frd.append(f); surf.append(np.full(len(el_r), self.n_surfaces, dtype=int))
        self.x = np.concatenate(X); self.y = np.concatenate(Y); self.z = np.concatenate(Z)
        self.conn_glob = np.vstack(conn); self.conn_surf = self.conn_glob
        self.boundary_conditions = bc
        self.frame_of_reference_delta = np.vstack(frd)
        self.surface_number = np.concatenate(surf)
        self.image_elem = (ne + 1, ne + 1 + len(el_r), el_r)
        self.image_map = {m: r for r, m in rmap.items()}           # nodo immagine -> nodo vero
        self.image_link_mid = iLm
        self.image_sign = -1.0 if os.environ['AC38_IMAGE'] == 'fs' else 1.0
        self.num_node_tot = len(self.x); self.num_elem_tot = len(self.conn_glob)
        self.n_surfaces += 1
        print("[immagine %s] piano z_A = %.4f (nodo %d), %d nodi, %d elementi"
              % (os.environ['AC38_IMAGE'], zf, i_f, len(real), len(el_r)))

    # ------------------------------------------------------------- mass/stiff
    def update_mass_stiff(self):
        """Timone dalle sezioni; elevatore con la 6x6 della punta della lama (di
        comodo: la struttura di SHARPy non entra nelle GAF proiettate)."""
        nr = self.n_rudder_elem
        ne_tot = self.num_elem_tot
        self.num_elem_tot = nr
        conn_full = self.conn_glob
        self.conn_glob = conn_full[:nr]
        super().update_mass_stiff()
        self.conn_glob = conn_full
        self.num_elem_tot = ne_tot
        Ks, Ms = self.stiffness[nr - 1], self.mass[nr - 1]
        self.stiffness = np.concatenate([self.stiffness, np.repeat(Ks[None], ne_tot - nr, 0)])
        self.mass = np.concatenate([self.mass, np.repeat(Ms[None], ne_tot - nr, 0)])
        self.elem_stiffness = np.arange(ne_tot, dtype=int)
        self.elem_mass = np.arange(ne_tot, dtype=int)

    # ------------------------------------------------------------------ aero
    def update_aero_prop(self):
        nr = self.n_rudder_elem
        ne_tot = self.num_elem_tot
        self.num_elem_tot = nr
        conn_full = self.conn_glob
        self.conn_glob = conn_full[:nr]
        nn_full = self.num_node_tot
        zn_r = self._zn
        super().update_aero_prop()           # timone (usa self._zn dei soli nodi del timone)
        chord_r, ea_r, aero_r = self.chord, self.elastic_axis, self.aero_node
        self.conn_glob = conn_full; self.num_elem_tot = ne_tot

        flat = np.column_stack(geo_utils.interpolate_naca_camber(0., 0., 0., 0., 0.))
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros((ne_tot, self.num_node_elem), dtype=int)
        self.surface_m = self.M * np.ones((self.n_surfaces,), dtype=int)
        self.twist = np.zeros((ne_tot, 3))
        self.chord = np.zeros((ne_tot, 3)); self.elastic_axis = np.zeros((ne_tot, 3))
        self.chord[:nr] = chord_r; self.elastic_axis[:nr] = ea_r

        E = self._elev
        # bordo d'attacco = x MASSIMO nel CAD (colonna 'x_TE_mm' del csv, che
        # era stata nominata al contrario: e' il bordo arrotondato)
        le = np.interp(self._ye, E['y'], E['te'])
        te = np.interp(self._ye, E['y'], E['le'])
        if os.environ.get('AC38_BUG_ELEV'):   # diagnostica: bordi invertiti (errato)
            le, te = te, le
        c = (le - te)*1e-3
        # posizione in corda della trave nel CAD: il centro di tensione vero
        x_beam = self._xtc_e
        ea = (le - x_beam)*1e-3 / c
        for lab in ('port', 'stbd'):
            e0, e1 = self.elev_elem_ranges[lab]
            g = self.elev_nodes[lab]
            for k, ie in enumerate(range(e0, e1)):
                loc = np.array([2*k, 2*k + 2, 2*k + 1])     # ordine [0, 2, 1] dei nodi
                if lab == 'port' and os.environ.get('AC38_ELEV_REVERSE'):
                    kk = len(range(e0, e1)) - 1 - k
                    loc = np.array([2*kk + 2, 2*kk, 2*kk + 1])
                self.chord[ie, :] = c[loc]
                self.elastic_axis[ie, :] = ea[loc]
        self.aero_node = np.ones(nn_full, dtype=bool)
        self.aero_node[:len(aero_r)] = aero_r
        if self.image_map:
            e0, e1, el_r = self.image_elem
            for ie, ir in zip(range(e0, e1), el_r):
                self.chord[ie, :] = chord_r[ir]; self.elastic_axis[ie, :] = ea_r[ir]
            for m, r in self.image_map.items():
                self.aero_node[m] = aero_r[r]
        self._zn = zn_r
        L = float(os.environ.get('AC38_WAKE_MATCH', '0') or 0.0) * 1e-3
        self.surface_group = np.arange(self.n_surfaces)          # superficie "logica" di ogni superficie SHARPy
        if L > 0:
            self._match_panels(L)

    def _match_panels(self, L, m_min=4, m_max=48, tol=0.15):
        """Scia 'corretta' (AC38_WAKE_MATCH = L [mm]): il passo di scia e' unico (U dt = L), quindi si porta la
        lunghezza dei pannelli in corda a ~L ovunque. Ogni superficie viene divisa lungo l'apertura in tratti di
        elementi consecutivi con lo stesso numero di pannelli in corda M = corda / L (entro tol); ogni tratto
        diventa una superficie SHARPy. Gli elementi senza pannelli (parte asciutta della lama) prendono la M
        dell'elemento bagnato piu' vicino. Banco su ala rettangolare: scia 3.3x piu' rada / fine del pannello
        -> errore ~100 % / ~25 % sul momento di beccheggio; scia = pannello -> ~6 % con 14 pannelli."""
        sn = self.surface_number
        new = np.full_like(sn, -1)
        m_list, group = [], []
        for s_ in range(self.n_surfaces):
            els = np.flatnonzero(sn == s_)
            if not len(els):
                continue
            wet = np.array([bool(np.all(self.aero_node[self.conn_glob[e]])) for e in els])
            c = self.chord[els].mean(axis=1)
            Me = np.clip(np.round(c / L), m_min, m_max)
            if wet.any() and not wet.all():
                iw = np.flatnonzero(wet)
                Me = Me[iw[np.argmin(np.abs(np.arange(len(els))[:, None] - iw[None, :]), axis=1)]]
                c = c[iw[np.argmin(np.abs(np.arange(len(els))[:, None] - iw[None, :]), axis=1)]]
            k0 = 0
            for k in range(1, len(els) + 1):
                if k == len(els) or abs(Me[k] - Me[k0]) / Me[k0] > tol:
                    new[els[k0:k]] = len(m_list)
                    m_list.append(int(np.clip(np.round(c[k0:k].mean() / L), m_min, m_max)))
                    group.append(s_)
                    k0 = k
        self.surface_number = new
        self.surface_m = np.array(m_list, dtype=int)
        self.surface_group = np.array(group, dtype=int)
        self.n_surfaces = len(m_list)
        print("[scia adattata] pannello/passo di scia %.1f mm: %d superfici, M = %s"
              % (1e3 * L, self.n_surfaces, m_list))
