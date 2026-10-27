"""Caso SHARPy del braccio ETNZ AC75 (examples/tnz_multibody) — superficie portante UVLM.

Come `ac37_tfoil.AC37TFoil` serve SOLO la superficie: le GAF si proiettano sui modi di GINO
(precompute_quvlm_etnz.py), la struttura di SHARPy e' di comodo.

GEOMETRIA (= griglia DLM del report LV del 24/03/2026, MULTI_tnz_boot_tnz_foil_dx_tnz_foil_sx):
  boot  : piano verticale y = 0, z in [-0.3, 0], corda 0.8, bordo d'attacco 340 mm avanti il CT del
          braccio (x = 0), fisso in tutte le configurazioni (TNZ: "0.8 m chord, 0.3 m span")
  ala   : due semiali dalla giunzione (0, 0, -0.3) a (0, +-1.879, -0.984): anedro 20 deg, 2 m per lato,
          corda 0.4, bordo d'attacco a x = TNZ_LE
Frame A di SHARPy = frame di GINO (x lungo la corrente, z verso l'alto). Il CT del braccio sta
sulla linea x = 0; i nodi dell'ala a meta' corda, collegamento senza pannelli dalla punta del
boot alla radice dell'ala. Pannelli adattati al passo di scia (ETNZ_WAKE_MATCH [mm], default 20).
"""
import os
import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils
from gino_bem.ac38_tfoil import AC38TFoil

RHO_W = 1025.0
C_BOOT, X_LE_BOOT, Z_TOP, Z_JUN = 0.8, -0.34, 0.0, -0.3
C_WING, L_WING, DIHEDRAL = 0.4, 2.0, np.deg2rad(20.0)
LE = float(os.environ.get("TNZ_LE", "-0.1"))
GRIDS = {
    # (N boot, N per semiala)
    "w20": (12, 30),
    "w15": (12, 40),
    "w15fine": (16, 60),        # controllo di convergenza in apertura
}
WAKE_MATCH = float(os.environ.get("ETNZ_WAKE_MATCH", "20")) * 1e-3


class ETNZTFoil(fw.FlyingWing):

    def __init__(self, N, N_wing, u_inf, route='.', case_name='etnz'):
        if N % 2 or N_wing % 2:
            raise ValueError("N e N_wing devono essere pari (elementi a 3 nodi)")
        self.N_wing = int(N_wing)
        M = int(round(C_WING / WAKE_MATCH))
        super().__init__(M=M, N=N, Mstar_fact=1, u_inf=u_inf, alpha=0.0, rho=RHO_W,
                         b_ref=Z_TOP - Z_JUN, main_chord=WAKE_MATCH * M, aspect_ratio=(Z_TOP - Z_JUN) / (WAKE_MATCH * M),
                         roll=0.0, beta=0.0, yaw=0.0, sweep=0.0, n_surfaces=1,
                         physical_time=2, route=route, case_name=case_name, RollNodes=False)

    def update_derived_params(self):
        """Topologia completa (boot + collegamento + 2 semiali) prima degli array di comodo."""
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
        z = np.linspace(Z_TOP, Z_JUN, nr)
        X, Y, Z = [np.zeros(nr)], [np.zeros(nr)], [z]
        conn = [np.array([conn_loc + 2 * i for i in range(self.N // 2)])]
        bc = -np.ones(nr, dtype=int); bc[0] = 1; bc[-1] = 0            # incastro / giunzione
        bcs = [bc]
        frd = [np.tile([-1.0, 0.0, 0.0], (self.N // 2, nne, 1))]
        surf = [np.zeros(self.N // 2, dtype=int)]
        self.n_boot_node, self.n_boot_elem, self.tip_idx = nr, self.N // 2, nr - 1
        nn, ne = nr, self.N // 2
        xw = LE + 0.5 * C_WING
        # collegamento (senza pannelli) punta del boot -> radice dell'ala a meta' corda
        iC, iCm = nn, nn + 1
        X.append(np.array([xw, 0.5 * xw])); Y.append(np.zeros(2)); Z.append(np.full(2, Z_JUN))
        conn.append(np.array([[self.tip_idx, iC, iCm]])); bcs.append(np.zeros(2, dtype=int))
        frd.append(np.tile([0.0, 0.0, 1.0], (1, nne, 1))); surf.append(np.array([-1]))
        self.link_elem, self.wing_root = ne, iC
        nn += 2; ne += 1
        s = np.linspace(0.0, L_WING, self.N_wing + 1)
        self.wing_s = s
        self.wing_nodes, self.wing_elem_ranges = {}, {}
        for sign, sid, lab in ((+1.0, 1, "foil_dx"), (-1.0, 2, "foil_sx")):
            g = np.empty(self.N_wing + 1, dtype=int); g[0] = iC; g[1:] = nn + np.arange(self.N_wing)
            X.append(np.full(self.N_wing, xw)); Y.append(sign * s[1:] * np.cos(DIHEDRAL))
            Z.append(Z_JUN - s[1:] * np.sin(DIHEDRAL))
            conn.append(np.array([g[conn_loc + 2 * i] for i in range(self.N_wing // 2)]))
            b = np.zeros(self.N_wing, dtype=int); b[-1] = -1
            bcs.append(b)
            frd.append(np.tile([-1.0, 0.0, 0.0], (self.N_wing // 2, nne, 1)))
            surf.append(np.full(self.N_wing // 2, sid, dtype=int))
            self.wing_nodes[lab] = g
            self.wing_elem_ranges[lab] = (ne, ne + self.N_wing // 2)
            nn += self.N_wing; ne += self.N_wing // 2
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
        c = np.full(self.num_node_tot, C_WING); ea = np.full(self.num_node_tot, 0.5)
        c[:self.n_boot_node] = C_BOOT
        ea[:self.n_boot_node] = (0.0 - X_LE_BOOT) / C_BOOT
        return c, ea

    def update_mass_stiff(self):
        # struttura di comodo (i modi sono quelli di GINO): sezione uniforme, ben condizionata
        ne = self.num_elem_tot
        self.stiffness = np.tile(np.diag([1e10, 1e9, 1e9, 1e8, 1e8, 1e9]), (ne, 1, 1))
        self.mass = np.tile(np.diag([50.0, 50.0, 50.0, 1.0, 0.5, 0.5]), (ne, 1, 1))
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
        self.surface_group = np.arange(self.n_surfaces)
        if WAKE_MATCH > 0:
            AC38TFoil._match_panels(self, WAKE_MATCH)
