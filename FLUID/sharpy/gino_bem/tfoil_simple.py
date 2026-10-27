"""Caso SHARPy del T-foil semplificato (examples/TFOIL_SIMPLE) — superficie portante UVLM.

Come `ac38_tfoil.AC38TFoil` serve SOLO la superficie: le GAF si proiettano sui modi
di GINO (precompute_quvlm_tfoil_simple.py), la struttura di SHARPy e' di comodo
(sezioni dello stesso modulo di geometria, nessun baricentro fuori asse).

GEOMETRIA da examples/TFOIL_SIMPLE/tfoil_simple_geom.py. Frame A di SHARPy: corrente
lungo +x_A; timone lungo +z_A (roll 90, z_A = y_G di GINO, 0 all'incastro);
elevatore dal nodo di punta lungo +-y_A (y_A = -z_G: la superficie 1 a +y_A e'
la semiala 'elevator_stbd' di GINO). Innesto a nodo condiviso (convenzione
Ttail_3beams: bc 0 sul nodo di giunzione).
"""
import importlib.util
import os
import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

_FSI = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
_s = importlib.util.spec_from_file_location(
    "tfoil_simple_geom", os.path.join(_FSI, "examples", "TFOIL_SIMPLE", "tfoil_simple_geom.py"))
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

GRIDS = {
    # (M corda, N timone, Mstar, N per semiala)
    "n40c12e16": (12, 40, 6, 16),
    "n40c16e16": (16, 40, 6, 16),     # controllo di convergenza in corda
    "n40c20e16": (20, 40, 6, 16),
    "coarse":    (8, 24, 6, 10),
}


def _sharpy_section(c):
    """K, M nell'ordine SHARPy [EA, GAy, GAz, GJ, EI_flap, EI_edge] (di comodo)."""
    K, M = G.section(c)
    k = np.diag([K[1, 1], K[0, 0], K[2, 2], K[4, 4], K[3, 3], K[5, 5]])
    m = np.diag([M[0, 0], M[0, 0], M[0, 0], M[4, 4], M[3, 3], M[5, 5]])
    return k, m


class TfoilSimple(fw.FlyingWing):

    def __init__(self, M, N, Mstar_fact, u_inf, N_elev=16, route='.', case_name='tfoil_simple'):
        if N % 2 or N_elev % 2:
            raise ValueError("N e N_elev devono essere pari (elementi a 3 nodi)")
        self.N_elev = int(N_elev)
        span = G.L_RUDDER
        # TFS_ALPHA [deg] (21/10/2026): incidenza di base per il controllo alpha = 0 contro alpha > 0 (default 0, come prima)
        super().__init__(M=M, N=N, Mstar_fact=Mstar_fact, u_inf=u_inf, alpha=float(os.environ.get("TFS_ALPHA", "0")), rho=G.RHO_W,
                         b_ref=span, main_chord=G.C_ROOT, aspect_ratio=span / G.C_ROOT,
                         roll=90.0, beta=0.0, yaw=0.0, sweep=0.0, n_surfaces=1,
                         physical_time=2, route=route, case_name=case_name, RollNodes=True)
        self.c_ref = G.C_ROOT

    def update_derived_params(self):
        """Topologia completa (timone + 2 semiali) prima degli array di comodo."""
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
        s = np.linspace(0.0, G.L_RUDDER, nr)
        # pelo libero sul timone: un nodo di ESTREMITA' d'elemento (indice pari) esattamente alla linea
        # d'acqua, almeno un elemento bagnato. Con passo fisso (65 mm) la parte bagnata "scattava" al nodo
        # piu' vicino e quote diverse davano la stessa tabella (bug 16/10/2026).
        s_wl = G.Z_ROOT - G.WL
        if 1e-9 < s_wl < G.L_RUDDER - 1e-9:
            j = int(2 * round(s_wl / (s[2] - s[0])))
            j = min(max(j, 2), nr - 3)
            s = np.concatenate([np.linspace(0.0, s_wl, j + 1), np.linspace(s_wl, G.L_RUDDER, nr - j)[1:]])
        X, Y, Z = [np.zeros(nr)], [np.zeros(nr)], [s]
        conn = [np.array([conn_loc + 2 * i for i in range(self.N // 2)])]
        bc = -np.ones(nr, dtype=int); bc[0] = 1; bc[-1] = 0          # incastro / giunto a T
        bcs = [bc]
        frd = [np.tile([-1.0, 0.0, 0.0], (self.N // 2, nne, 1))]
        surf = [np.zeros(self.N // 2, dtype=int)]
        self.rudder_s = s
        self.n_rudder_node, self.n_rudder_elem = nr, self.N // 2
        tip = nr - 1
        self.tip_idx = tip
        nn, ne = nr, self.N // 2
        se = np.linspace(0.0, G.ELEV_SEMISPAN, self.N_elev + 1)
        self.elev_s = se
        self.elev_nodes, self.elev_elem_ranges = {}, {}
        edx = float(getattr(G, "EDX", 0.0))
        root = tip
        self.link_elem = None
        if edx:
            # collegamento (senza pannelli) punta del timone -> radice dell'elevatore spostata in x
            iC, iCm = nn, nn + 1
            X.append(np.array([edx, 0.5 * edx])); Y.append(np.zeros(2)); Z.append(np.full(2, G.L_RUDDER))
            conn.append(np.array([[tip, iC, iCm]])); bcs.append(np.zeros(2, dtype=int))
            frd.append(np.tile([-1.0, 0.0, 0.0], (1, nne, 1))); surf.append(np.array([-1]))
            self.link_elem = ne; root = iC
            nn += 2; ne += 1
        self.elev_root = root
        for sign, sid, lab in ((+1.0, 1, "elevator_stbd"), (-1.0, 2, "elevator_port")):
            g = np.empty(self.N_elev + 1, dtype=int); g[0] = root; g[1:] = nn + np.arange(self.N_elev)
            X.append(np.full(self.N_elev, edx)); Y.append(sign * se[1:]); Z.append(np.full(self.N_elev, G.L_RUDDER))
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

    def _node_chord(self):
        c = np.zeros(self.num_node_tot)
        c[:self.n_rudder_node] = G.chord_rudder(-(G.Z_ROOT - self.rudder_s))
        for lab, g in self.elev_nodes.items():
            c[g] = G.chord_elev(self.elev_s)
        if self.link_elem is not None:
            c[self.conn_glob[self.link_elem][2]] = G.chord_elev(0.0)
        return c

    def update_mass_stiff(self):
        cn = self._node_chord()
        ne = self.num_elem_tot
        self.stiffness = np.zeros((ne, 6, 6)); self.mass = np.zeros((ne, 6, 6))
        for ie in range(ne):
            k, m = _sharpy_section(cn[self.conn_glob[ie]].mean())
            self.stiffness[ie], self.mass[ie] = k, m
        self.elem_stiffness = np.arange(ne, dtype=int)
        self.elem_mass = np.arange(ne, dtype=int)

    # ------------------------------------------------------------------ aero
    def update_aero_prop(self):
        flat = np.column_stack(geo_utils.interpolate_naca_camber(0., 0., 0., 0., 0.))
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros((self.num_elem_tot, self.num_node_elem), dtype=int)
        self.surface_m = self.M * np.ones((self.n_surfaces,), dtype=int)
        self.twist = np.zeros((self.num_elem_tot, 3))
        cn = self._node_chord()
        self.chord = cn[self.conn_glob]
        self.elastic_axis = np.full((self.num_elem_tot, 3), G.XEA)
        self.aero_node = np.ones(self.num_node_tot, dtype=bool)
        # timone: pannelli solo sotto il pelo libero (nodo piu' vicino alla linea d'acqua)
        z_phys = G.Z_ROOT - self.rudder_s
        self.aero_node[:self.n_rudder_node] = z_phys <= min(G.WL, G.Z_ROOT) + 1e-9   # nodo alla linea d'acqua (vedi sopra)
