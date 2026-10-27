"""
Besch & Liu (1973) Model 2T strut, config G — SHARPy FlyingWing template
(stock UVLM, rho water).

Geometry / structure aligned with GINO ``examples/Model2t/config_Model2T.py``:
  span 1.2622 m along the 15deg-swept EA, chord 0.2944 m, cantilever clamped
  at the (dry) root, rolled vertical. Only the span beyond 0.254 m from the
  root is wetted -> no aero panels on the dry root segment (aero_node mask).

Tip pod (config G) enters STRUCTURALLY only (lumped mass + inertia + CG
offset). Stock UVLM is a lifting-surface method: it cannot represent the
pod's hydrodynamic added mass, nor the free surface. Same limitation class
as the Abramson UVLM run (see SHARPY_UVLM_ANALYSIS.md).

Structural stiffness defaults to the GINO headline calibration
(GJ-only factor 0.4953, see memory/model2t_stiffness_calibration).
"""

from __future__ import annotations

import numpy as np

import sharpy.cases.templates.flying_wings as fw
import sharpy.utils.algebra as algebra
import sharpy.utils.geo_utils as geo_utils

IN2M = 0.0254

# --- Besch & Liu Table 1 / Fig. 3 / Fig. 20 (config G), as in config_Model2T.py ---
SPAN_EA = 49.693 * IN2M          # [m] 1.2622, along the 15deg-swept EA
CHORD = 5.796 * 2 * IN2M         # [m] 0.29444
SWEEP_DEG = 15.0
WETTED_FROM_ROOT = 10.0 * IN2M   # [m] 0.254 arc length along EA above waterline (dry)
MU = 18.8                        # [kg/m]  0.00273 lb s^2/in^2
I_ALPHA = 0.16812                # [kg m^2/m] torsion inertia about EA
EIXX = 3710.91                   # [N m^2] flapwise (1.293e6 lbf in^2)
GJ = 5940.9                      # [N m^2] (2.070e6 lbf in^2)
XEA = 0.69                       # shear center at 69 % chord from LE (a=+0.38)
XCM = 0.583                      # CG at 58.3 % chord from LE (x_alpha=-0.214)

# Tip pod, config G (structural spheroid uses BODY length 25.7 in — the GJ/EI
# calibrations were done with this inertia; do not switch to the 37.7 in total).
POD_MASS = 38.4 * 0.45359237     # [kg] 17.42
POD_BODY_LEN = 25.7 * IN2M       # [m]
POD_DIAM = 2.0 * IN2M            # [m]
POD_CG_AFT = 7.8 * IN2M          # [m] pod CG aft of the strut EA
_A_POD = POD_BODY_LEN / 2.0
_B_POD = POD_DIAM / 2.0
I_POD_AXIAL = (2.0 / 5.0) * POD_MASS * _B_POD**2            # about pod long axis (chord)
I_POD_TRANS = (1.0 / 5.0) * POD_MASS * (_A_POD**2 + _B_POD**2)

# GINO headline stiffness calibration (wet-1T match with 37.7-in pod BEM)
GJ_FACTOR_HEADLINE = 0.4953


class Model2tStrut(fw.FlyingWing):
    """Single-surface swept cantilever strut with structural tip pod."""

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
        case_name="model2t",
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
            sweep=SWEEP_DEG,
            n_surfaces=1,
            physical_time=2,
            route=route,
            case_name=case_name,
            RollNodes=True,
        )
        self.span = float(span)
        self.c_ref = float(main_chord)
        self.main_ea = XEA
        self.main_cg = XCM
        self.sigma = 1.0
        self.gj_factor = float(gj_factor)
        self.ei_factor = float(ei_factor)
        self.wetted_from_root = WETTED_FROM_ROOT

    def update_fem_prop(self):
        """Cantilever along the swept EA (pre-roll in xy); root clamped at node 0."""
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

        # Arc length along the swept EA; sweep tilts the axis downstream (+x).
        s = np.linspace(0.0, self.wing_span, num_node_tot)
        x = np.sin(self.sweep_rad) * s
        y = np.cos(self.sweep_rad) * s
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

    def update_aero_prop(self):
        """
        Aero props for the full-span cantilever, then blank the dry root span
        (above the waterline). The template single-surface branch assumes the
        mirrored half-wing layout (even element count), so build the airfoil
        distribution here: one flat-plate profile on every element (the blunt
        strut section is symmetric — camber only in linear UVLM).
        """
        flat = np.column_stack(
            geo_utils.interpolate_naca_camber(0.0, 0.0, 0.0, 0.0, 0.0)
        )
        self.Airfoils_surf = [flat]
        self.airfoil_distribution = np.zeros(
            (self.num_elem_tot, self.num_node_elem), dtype=int
        )

        self.surface_m = self.M * np.ones((self.n_surfaces,), dtype=int)
        self.twist = np.zeros((self.num_elem_tot, 3))
        self.chord = self.main_chord * np.ones((self.num_elem_tot, 3))
        self.elastic_axis = self.main_ea * np.ones((self.num_elem_tot, 3))

        s = np.linspace(0.0, self.wing_span, self.num_node_tot)
        ds = s[1] - s[0]
        # Nearest-node waterline: node at s=0.2524 (N multiple of 10) is wet.
        self.aero_node = s >= (self.wetted_from_root - 0.5 * ds)

    def update_mass_stiff(self):
        """
        GINO section values on the SHARPy B-frame diag
        [EA, GAy, GAz, GJ, EI_y(flap), EI_z(lag)] with xb along the swept EA.
        """
        chord = self.main_chord

        ea = EIXX * 1e3
        ga = GJ * 1e3
        gj = self.gj_factor * GJ
        eiy = self.ei_factor * EIXX
        eiz = EIXX * 1e4

        base_stiffness = np.diag([ea, ga, ga, self.sigma * gj, eiy, eiz])
        self.stiffness = np.zeros((1, 6, 6))
        self.stiffness[0] = base_stiffness

        # CG offset chordwise (positive aft, Goland/Abramson convention):
        # CG forward of EA for Model 2T -> e < 0.
        e = (self.main_cg - self.main_ea) * chord
        # I_ALPHA is about the EA; split bending rotary inertias about the CG.
        i_cg = max(I_ALPHA - MU * e**2, 0.0)
        i22 = 0.3 * i_cg
        i33 = 0.7 * i_cg + MU * e**2

        m_chi = algebra.skew(MU * np.array([0.0, e, 0.0]))
        Msec = np.diag([MU, MU, MU, I_ALPHA, i22, i33])
        Msec[:3, 3:] = m_chi
        Msec[3:, :3] = -m_chi

        self.mass = np.zeros((1, 6, 6))
        self.mass[0] = Msec

        self.elem_stiffness = np.zeros((self.num_elem_tot,), dtype=int)
        self.elem_mass = np.zeros((self.num_elem_tot,), dtype=int)

        # Tip pod as lumped mass at the free end. B-frame of the tip element:
        # xb outboard along the swept EA, yb points to the LE (aft = -yb; the
        # distributed +skew(mu*[0,e,0]) coupling above is the template's
        # equivalent of SHARPy's lumped -m*skew(r) with r_y = -e). Pod centre
        # hangs b_pod beyond the tip, CG 7.8 in aft of the EA -> r_y < 0.
        # Inertia about the pod CG: long axis is streamwise (yb) -> I_axial
        # on yy, transverse on xx/zz; parallel-axis terms are assembled by
        # SHARPy from the position offset (generate_mass_matrix).
        self.lumped_mass = np.array([POD_MASS])
        self.lumped_mass_nodes = np.array([self.num_node_tot - 1], dtype=int)
        self.lumped_mass_position = np.array([[_B_POD, -POD_CG_AFT, 0.0]])
        self.lumped_mass_inertia = np.array(
            [np.diag([I_POD_TRANS, I_POD_AXIAL, I_POD_TRANS])]
        )
