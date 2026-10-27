"""
Abramson 1965 hydrofoil — SHARPy FlyingWing template (stock UVLM, rho water).

Geometry / structure aligned with GINO ``examples/ABRAMSON1965/config_ABRAMSON1965.py``:
  span 0.762 m, chord 0.3048 m, cantilever clamped at root, rolled vertical (span ~ +Z).
"""

from __future__ import annotations

import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.algebra as algebra


class Abramson1965(fw.FlyingWing):
    """Single-surface cantilever; properties from GINO Abramson config."""

    def __init__(
        self,
        M,
        N,
        Mstar_fact,
        u_inf,
        alpha=0.0,
        rho=997.0,
        main_chord=0.3048,
        span=0.762,
        roll=90.0,
        route=".",
        case_name="abramson1965",
    ):
        aspect_ratio = span / main_chord
        super().__init__(
            M=M,
            N=N,
            Mstar_fact=Mstar_fact,
            u_inf=u_inf,
            alpha=alpha,
            rho=rho,
            b_ref=span,
            main_chord=main_chord,
            aspect_ratio=aspect_ratio,
            roll=roll,
            beta=0.0,
            yaw=0.0,
            sweep=0.0,
            n_surfaces=1,
            physical_time=2,
            route=route,
            case_name=case_name,
            RollNodes=True,
        )
        self.span = float(span)
        self.c_ref = float(main_chord)
        self.main_ea = 0.25
        self.main_cg = 0.512
        self.sigma = 1.0

    def update_fem_prop(self):
        """Cantilever along +Y (pre-roll); root clamped at node 0."""
        num_node_elem = self.num_node_elem
        num_elem_tot = self.N // 2
        num_node_tot = self.N + 1
        self.num_elem_tot = num_elem_tot
        self.num_elem_surf = num_elem_tot
        self.num_node_surf = num_node_tot
        self.num_node_tot = num_node_tot

        conn_loc = np.array([0, 2, 1], dtype=int)
        conn_glob = np.zeros((num_elem_tot, num_node_elem), dtype=int)
        for ielem in range(num_elem_tot):
            conn_glob[ielem, :] = conn_loc + ielem * (num_node_elem - 1)

        y = np.linspace(0.0, self.wing_span, num_node_tot)
        x = np.zeros_like(y)
        z = np.zeros_like(y)

        if self.RollNodes:
            sr, cr = np.sin(self.roll_rad), np.cos(self.roll_rad)
            yold = y.copy()
            y = cr * yold
            z = sr * yold

        surface_number = np.zeros((num_elem_tot,), dtype=int)
        boundary_conditions = -np.ones((num_node_tot,), dtype=int)
        boundary_conditions[0] = 1

        frame_of_reference_delta = np.zeros((num_elem_tot, num_node_elem, 3))
        for ielem in range(num_elem_tot):
            for inode in range(num_node_elem):
                frame_of_reference_delta[ielem, inode, :] = [-1.0, 0.0, 0.0]

        self.frame_of_reference_delta = frame_of_reference_delta
        self.boundary_conditions = boundary_conditions
        self.conn_glob = conn_glob
        self.conn_surf = conn_glob
        self.surface_number = surface_number
        self.x = x
        self.y = y
        self.z = z

    def update_mass_stiff(self):
        """
        Map GINO section matrices (beam Y = span) to SHARPy B-frame diag:
        [EA, GA, GA, GJ, EI_y, EI_z] with xb along span.
        """
        chord = self.main_chord
        mu = 72.14
        ea = 1.0e8
        ga = 1.0e8
        gj = 2.79e3
        eiy = 9.76e3
        eiz = 1.0e9

        base_stiffness = np.diag([ea, ga, ga, self.sigma * gj, eiy, eiz])
        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = base_stiffness

        semichord = 0.5 * chord
        r2_dim = 0.512 * semichord**2
        i11 = mu * r2_dim
        i22 = 0.3 * i11
        i33 = 0.7 * i11
        e = (self.main_cg - self.main_ea) * chord

        # B-frame (Goland): DOF 3 = torsion θ_xb, 4 = flap θ_yb, 5 = chord θ_zb.
        # GINO section M uses [θx, θy, θz] → permute inertias 3↔4; stiffness already B-frame.
        m_chi = algebra.skew(mu * np.array([0.0, e, 0.0]))
        Msec = np.diag([mu, mu, mu, i11 + mu * e**2, i22, i33 + mu * e**2])
        Msec[:3, 3:] = m_chi
        Msec[3:, :3] = -m_chi

        self.mass = np.zeros((1, 6, 6))
        self.mass[0] = Msec

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)

        self.lumped_mass = np.zeros((1,))
        self.lumped_mass_position = np.zeros((1, 3))
        self.lumped_mass_inertia = np.zeros((1, 3, 3))
        self.lumped_mass_nodes = np.zeros((1,), dtype=int)
