"""
Model 2T T-foil test case -- SHARPy FlyingWing-derived template (stock UVLM).

Vertical strut + tip pod: IDENTICAL to ``model2t_strut.Model2tStrut`` (same
geometry, section stiffness/mass, pod). New horizontal foils are appended at
the strut TIP node (shared/branch node, same T-junction pattern as SHARPy's
own ``sharpy.cases.templates.Ttail.Ttail_3beams`` template): semi-span =
strut span / 4, chord = strut chord / 4 (so AR matches the strut's), 0 deg
sweep, and the SAME section stiffness/mass matrices as the strut (literal
reuse -- mirrors ``STRUCTURE/FEA/structural_model.py``'s
``foil_reuse_strut_section`` on the GINO side).

``FlyingWing`` (the base template) hard-caps ``n_surfaces < 3`` and assumes a
single planar (or mirrored) wing, so it cannot build a T-junction directly.
This class works around that by subclassing ``Model2tStrut`` (n_surfaces=1,
satisfying the base assert) and APPENDING the two foil branches on top of the
already-built strut arrays in ``update_fem_prop``/``update_aero_prop``, then
correcting ``num_node_tot``/``num_elem_tot`` and the few strut-only-sized
arrays (``control_surface``) that the base ``update_derived_params`` sizes
before the branches exist.

Boundary-condition convention for the T-junction node follows Ttail_3beams:
0 (ordinary/continuation node, NOT a free end) once the foils are attached --
the strut's own ``update_fem_prop`` marks it -1 (true free tip) because in
isolation it is one; that value is overwritten here.
"""

from __future__ import annotations

import numpy as np

import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

from gino_bem.model2t_strut import (
    CHORD,
    GJ_FACTOR_HEADLINE,
    SPAN_EA,
    Model2tStrut,
)


class Model2tTfoilStrut(Model2tStrut):
    """Model2t strut + tip pod + 2 horizontal foils (inverted-T)."""

    def __init__(
        self,
        M,
        N,
        Mstar_fact,
        u_inf,
        alpha=0.0,
        rho=997.0,
        main_chord=CHORD,
        span=SPAN_EA,
        roll=90.0,
        gj_factor=GJ_FACTOR_HEADLINE,
        ei_factor=1.0,
        route=".",
        case_name="model2t_tfoil",
        N_foil=8,
        foil_semi_span=None,
        foil_chord=None,
        foil_sweep_deg=0.0,
    ):
        if N_foil % 2:
            raise ValueError(f"N_foil must be even (3-noded FEs), got {N_foil}")
        self.N_foil = int(N_foil)
        self.foil_semi_span = float(foil_semi_span if foil_semi_span is not None else span / 4.0)
        self.foil_chord = float(foil_chord if foil_chord is not None else main_chord / 4.0)
        self.foil_sweep_deg = float(foil_sweep_deg)
        super().__init__(
            M, N, Mstar_fact, u_inf, alpha=alpha, rho=rho, main_chord=main_chord,
            span=span, roll=roll, gj_factor=gj_factor, ei_factor=ei_factor,
            route=route, case_name=case_name,
        )

    # ------------------------------------------------------------------ params
    def update_derived_params(self):
        """Bypass FlyingWing's n_surfaces<3-sized bookkeeping; build the full
        (strut + 2 foils) topology first, THEN size the strut-only arrays
        (control_surface) that the base class would otherwise size too early."""
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
        """Strut exactly as Model2tStrut, then 2 foil branches off the tip node."""
        super().update_fem_prop()  # strut-only arrays; self.num_node_tot/num_elem_tot = strut-only for now

        tip_idx = self.num_node_tot - 1
        tip_pos = np.array([self.x[tip_idx], self.y[tip_idx], self.z[tip_idx]], dtype=float)
        # T-junction: no longer a free end (Ttail_3beams convention: 0, not -1).
        self.boundary_conditions[tip_idx] = 0

        self.strut_num_node = self.num_node_tot
        self.strut_num_elem = self.num_elem_tot
        self.tip_idx = tip_idx

        n_elem_foil = self.N_foil // 2
        conn_loc = np.array([0, 2, 1], dtype=int)
        tan_sweep = np.tan(np.deg2rad(self.foil_sweep_deg))

        x_parts, y_parts, z_parts = [self.x], [self.y], [self.z]
        conn_parts = [self.conn_glob]
        bc_parts = [self.boundary_conditions]
        frd_parts = [self.frame_of_reference_delta]
        surf_parts = [self.surface_number]

        node_next = self.num_node_tot
        elem_next = self.num_elem_tot
        self.foil_elem_ranges = {}
        self.foil_tip_node = {}

        for sign, surf_id, label in ((+1.0, 1, "port"), (-1.0, 2, "stbd")):
            s = np.linspace(0.0, self.foil_semi_span, self.N_foil + 1)  # local flat 0..N_foil
            flat_global = np.empty(self.N_foil + 1, dtype=int)
            flat_global[0] = tip_idx                      # shared root = T-junction
            flat_global[1:] = node_next + np.arange(self.N_foil)

            x_new = tip_pos[0] + sign * s[1:] * tan_sweep
            y_new = np.full(self.N_foil, tip_pos[1])
            z_new = tip_pos[2] + sign * s[1:]

            conn_foil = np.zeros((n_elem_foil, 3), dtype=int)
            for ie in range(n_elem_foil):
                conn_foil[ie, :] = flat_global[conn_loc + ie * 2]

            bc_new = np.zeros(self.N_foil, dtype=int)
            bc_new[-1] = -1  # outer tip = free end

            frd_new = np.zeros((n_elem_foil, self.num_node_elem, 3))
            frd_new[:, :, :] = np.array([-1.0, 0.0, 0.0])

            surf_new = np.full(n_elem_foil, surf_id, dtype=int)

            self.foil_elem_ranges[label] = (elem_next, elem_next + n_elem_foil)
            self.foil_tip_node[label] = int(flat_global[-1])

            x_parts.append(x_new); y_parts.append(y_new); z_parts.append(z_new)
            conn_parts.append(conn_foil)
            bc_parts.append(bc_new)
            frd_parts.append(frd_new)
            surf_parts.append(surf_new)

            node_next += self.N_foil
            elem_next += n_elem_foil

        self.x = np.concatenate(x_parts)
        self.y = np.concatenate(y_parts)
        self.z = np.concatenate(z_parts)
        self.conn_glob = np.vstack(conn_parts)
        self.conn_surf = self.conn_glob
        self.boundary_conditions = np.concatenate(bc_parts)
        self.frame_of_reference_delta = np.vstack(frd_parts)
        self.surface_number = np.concatenate(surf_parts)

        self.num_node_tot = len(self.x)
        self.num_elem_tot = len(self.conn_glob)

    # ------------------------------------------------------------------ mass/stiff
    def update_mass_stiff(self):
        super().update_mass_stiff()
        # Model2tStrut.update_mass_stiff placed the pod at `self.num_node_tot - 1`,
        # but by this point num_node_tot has already been overwritten (update_fem_prop
        # ran first) to the FULL strut+foils node count. Fix the pod back onto the
        # actual T-junction / strut tip node.
        self.lumped_mass_nodes = np.array([self.tip_idx], dtype=int)
        # elem_stiffness/elem_mass are already sized to the FULL element count here
        # (self.num_elem_tot was already updated) -- all elements (strut AND foils)
        # reference section 0, i.e. literal reuse of the strut's own K/M (same as
        # structural_model.py's foil_reuse_strut_section=True on the GINO side).

    # ------------------------------------------------------------------ aero
    def update_aero_prop(self):
        flat = np.column_stack(
            geo_utils.interpolate_naca_camber(0.0, 0.0, 0.0, 0.0, 0.0)
        )
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros(
            (self.num_elem_tot, self.num_node_elem), dtype=int
        )

        self.surface_m = self.M * np.ones((3,), dtype=int)
        self.twist = np.zeros((self.num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((self.num_elem_tot, 3))

        self.chord = np.full((self.num_elem_tot, 3), self.main_chord, dtype=float)
        for label in ("port", "stbd"):
            e0, e1 = self.foil_elem_ranges[label]
            self.chord[e0:e1, :] = self.foil_chord

        # Strut wetted mask (as Model2tStrut); foils sit entirely at the (deep)
        # strut-tip station -> fully wetted.
        s_strut = np.linspace(0.0, self.wing_span, self.strut_num_node)
        ds = s_strut[1] - s_strut[0]
        aero_node = np.ones((self.num_node_tot,), dtype=bool)
        aero_node[: self.strut_num_node] = s_strut >= (self.wetted_from_root - 0.5 * ds)
        self.aero_node = aero_node
