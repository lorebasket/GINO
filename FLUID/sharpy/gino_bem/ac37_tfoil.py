"""Caso SHARPy del timone AC37 (examples/AC37_TFOIL) — superficie portante UVLM.

Come `tfoil_simple.TfoilSimple` serve SOLO la superficie: le GAF si proiettano sui modi di
GINO (precompute_quvlm_ac37.py), la struttura di SHARPy e' di comodo.

GEOMETRIA da examples/AC37_TFOIL/ac37_geom.py. Frame A di SHARPy: corrente lungo +x_A;
timone lungo +z_A (z_A = y_G di GINO, 0 alla boccola inferiore); elevatore lungo +-y_A
(y_A = -z_G: la superficie a +y_A e' la semiala 'elevator_stbd' di GINO). I nodi stanno
sulle stesse linee dei centri di taglio della trave di GINO; corda e bordo d'attacco
dalla tabella, quindi elastic_axis varia da nodo a nodo.

Il timone porta pannelli solo sotto il pelo libero (G.Z_WL, immersione AC37_IMM sopra il
piano dell'elevatore, come le forze idro di ASWING sul metro inferiore) e arriva fino al
piano dell'elevatore (G.Y_SURF): nessun varco alla giunzione. Collegamento senza pannelli
dalla punta del timone alla radice dell'elevatore.

Pannelli adattati al passo di scia (AC37_WAKE_MATCH [mm], default 12.5, metodo di
AC38TFoil._match_panels): l'elevatore va da 299 a 35 mm di corda, con M costante il
pannello in punta sarebbe 8 volte piu' corto del passo di scia.
"""
import importlib.util
import os
import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils
from gino_bem.ac38_tfoil import AC38TFoil

_FSI = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
_s = importlib.util.spec_from_file_location(
    "ac37_geom", os.path.join(_FSI, "examples", "AC37_TFOIL", "ac37_geom.py"))
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

GRIDS = {
    # (M corda di riferimento, N timone, Mstar [file di scia], N per semiala)
    "n56c16e18": (16, 56, 64, 18),
    "n84c16e24": (16, 84, 64, 24),      # controllo di convergenza in apertura
    "coarse":    (8, 28, 32, 10),
}
WAKE_MATCH = float(os.environ.get("AC37_WAKE_MATCH", "12.5")) * 1e-3


def _sharpy_section(p):
    """K, M nell'ordine SHARPy [EA, GAy, GAz, GJ, EI_flap, EI_edge] (di comodo)."""
    K, M = G.section(p, 1.0)
    k = np.diag([K[1, 1], K[0, 0], K[2, 2], K[4, 4], K[3, 3], K[5, 5]])
    m = np.diag([M[0, 0], M[0, 0], M[0, 0], M[4, 4], M[3, 3], M[5, 5]])
    return k, m


class AC37TFoil(fw.FlyingWing):

    def __init__(self, M, N, Mstar_fact, u_inf, N_elev=18, route='.', case_name='ac37'):
        if N % 2 or N_elev % 2:
            raise ValueError("N e N_elev devono essere pari (elementi a 3 nodi)")
        self.N_elev = int(N_elev)
        span = G.Y_SURF - G.Y_TOP
        # main_chord / M = passo di scia (U = 1): con WAKE_MATCH il riferimento e' il pannello
        c_ref = WAKE_MATCH * M if WAKE_MATCH > 0 else 0.2
        super().__init__(M=M, N=N, Mstar_fact=Mstar_fact, u_inf=u_inf, alpha=0.0, rho=G.RHO_W,
                         b_ref=span, main_chord=c_ref, aspect_ratio=span / c_ref,
                         roll=90.0, beta=0.0, yaw=0.0, sweep=0.0, n_surfaces=1,
                         physical_time=2, route=route, case_name=case_name, RollNodes=True)
        self.c_ref = c_ref

    def update_derived_params(self):
        """Topologia completa (timone + collegamento + 2 semiali) prima degli array di comodo."""
        self.dt = self.main_chord / self.M / self.u_inf * self.dt_factor
        self.n_tstep = int(np.round(self.physical_time / self.dt))
        for a in ("alpha", "roll", "beta", "sweep", "yaw"):
            setattr(self, a + "_rad", getattr(self, a) * np.pi / 180.0)
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
        nne = self.num_node_elem
        conn_loc = np.array([0, 2, 1], dtype=int)
        nr = self.N + 1
        y = np.linspace(G.Y_TOP, G.Y_SURF, nr)
        # pelo libero su un nodo di ESTREMITA' d'elemento (indice pari), come tfoil_simple
        if G.Y_TOP + 1e-9 < G.Z_WL < G.Y_SURF - 1e-9:
            j = int(2 * round((G.Z_WL - G.Y_TOP) / (y[2] - y[0])))
            j = min(max(j, 2), nr - 3)
            y = np.concatenate([np.linspace(G.Y_TOP, G.Z_WL, j + 1), np.linspace(G.Z_WL, G.Y_SURF, nr - j)[1:]])
        X, Y, Z = [G.rudder_axis(y)], [np.zeros(nr)], [y]
        conn = [np.array([conn_loc + 2 * i for i in range(self.N // 2)])]
        bc = -np.ones(nr, dtype=int); bc[0] = 1; bc[-1] = 0          # incastro / giunto a T
        bcs = [bc]
        frd = [np.tile([-1.0, 0.0, 0.0], (self.N // 2, nne, 1))]
        surf = [np.zeros(self.N // 2, dtype=int)]
        self.rudder_y = y
        self.n_rudder_node, self.n_rudder_elem = nr, self.N // 2
        tip = nr - 1
        self.tip_idx = tip
        nn, ne = nr, self.N // 2
        P0, Ye = G.elev_nodes(self.N_elev, -1.0)                     # semiala stbd (z_G < 0 -> y_A > 0)
        xr, yr = P0[0, 0], P0[0, 1]
        self.elev_Y = Ye
        self.elev_nodes, self.elev_elem_ranges = {}, {}
        root = tip
        self.link_elem = None
        if np.hypot(xr - X[0][-1], yr - y[-1]) > 2e-3:
            # collegamento (senza pannelli) punta del timone -> radice dell'elevatore
            iC, iCm = nn, nn + 1
            X.append(np.array([xr, 0.5 * (xr + X[0][-1])])); Y.append(np.zeros(2)); Z.append(np.array([yr, 0.5 * (yr + y[-1])]))
            conn.append(np.array([[tip, iC, iCm]])); bcs.append(np.zeros(2, dtype=int))
            frd.append(np.tile([-1.0, 0.0, 0.0], (1, nne, 1))); surf.append(np.array([-1]))
            self.link_elem = ne; root = iC
            nn += 2; ne += 1
        self.elev_root = root
        for sign, sid, lab in ((+1.0, 1, "elevator_stbd"), (-1.0, 2, "elevator_port")):
            g = np.empty(self.N_elev + 1, dtype=int); g[0] = root; g[1:] = nn + np.arange(self.N_elev)
            X.append(P0[1:, 0]); Y.append(sign * Ye[1:]); Z.append(P0[1:, 1])
            conn.append(np.array([g[conn_loc + 2 * i] for i in range(self.N_elev // 2)]))
            b = np.zeros(self.N_elev, dtype=int); b[-1] = -1
            bcs.append(b)
            frd.append(np.tile([-1.0, 0.0, 0.0], (self.N_elev // 2, nne, 1)))
            surf.append(np.full(self.N_elev // 2, sid, dtype=int))
            self.elev_nodes[lab] = g
            self.elev_elem_ranges[lab] = (ne, ne + self.N_elev // 2)
            nn += self.N_elev; ne += self.N_elev // 2
        self.x, self.y, self.z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
        self.conn_glob = np.vstack(conn); self.conn_surf = self.conn_glob
        self.boundary_conditions = np.concatenate(bcs)
        self.frame_of_reference_delta = np.vstack(frd)
        self.surface_number = np.concatenate(surf)
        self.num_node_tot, self.num_elem_tot = len(self.x), len(self.conn_glob)
        self.num_node_surf, self.num_elem_surf = nr, self.N // 2
        self.n_surfaces = 3

    def _node_chord_ea(self):
        """Corda e posizione del nodo in frazione di corda dal bordo d'attacco, per nodo."""
        c = np.zeros(self.num_node_tot); ea = np.zeros(self.num_node_tot)
        yr = self.rudder_y
        c[:self.n_rudder_node] = G.chord_rudder(yr)
        ea[:self.n_rudder_node] = (G.rudder_axis(yr) - G.x_le_rudder(yr)) / c[:self.n_rudder_node]
        for lab, g in self.elev_nodes.items():
            c[g] = G.chord_elev(self.elev_Y)
            ea[g] = (self.x[g] - G.x_le_elev(self.elev_Y)) / c[g]
        if self.link_elem is not None:
            i = self.conn_glob[self.link_elem][2]
            c[i] = c[self.elev_root]; ea[i] = (self.x[i] - G.x_le_elev(0.0)) / c[i]
        return c, ea

    def update_mass_stiff(self):
        ne = self.num_elem_tot
        self.stiffness = np.zeros((ne, 6, 6)); self.mass = np.zeros((ne, 6, 6))
        p_r = G.rudder_props(1.0)
        for ie in range(ne):
            n = self.conn_glob[ie]
            if ie < self.n_rudder_elem:
                p = G.rudder_props(float(min(self.z[n].mean(), G.Y_END)))
            elif self.link_elem is not None and ie == self.link_elem:
                p = p_r
            else:
                p = G.elev_props(float(np.abs(self.y[n]).mean()))
            p = dict(p, e=0.0, mu=max(p["mu"], 1.0))      # l'ultima stazione della tabella ha peso 0
            self.stiffness[ie], self.mass[ie] = _sharpy_section(p)
        self.elem_stiffness = np.arange(ne, dtype=int)
        self.elem_mass = np.arange(ne, dtype=int)

    # ------------------------------------------------------------------ aero
    def update_aero_prop(self):
        flat = np.column_stack(geo_utils.interpolate_naca_camber(0., 0., 0., 0., 0.))
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros((self.num_elem_tot, self.num_node_elem), dtype=int)
        self.surface_m = self.M * np.ones((self.n_surfaces,), dtype=int)
        self.twist = np.zeros((self.num_elem_tot, 3))
        cn, ean = self._node_chord_ea()
        self.chord = cn[self.conn_glob]
        self.elastic_axis = ean[self.conn_glob]
        self.aero_node = np.ones(self.num_node_tot, dtype=bool)
        # timone: pannelli solo sotto il pelo libero (nodo alla linea d'acqua, vedi update_fem_prop)
        self.aero_node[:self.n_rudder_node] = self.rudder_y >= G.Z_WL - 1e-9
        self.surface_group = np.arange(self.n_surfaces)
        if WAKE_MATCH > 0:
            AC38TFoil._match_panels(self, WAKE_MATCH)
