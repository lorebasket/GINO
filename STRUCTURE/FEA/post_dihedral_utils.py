"""
Post-build rigid rotation about global +X by structural dihedral (deg).

Workflow: build beam, global K/M, modal data, and load aerogrid in the
reference frame (beam along +Y). Then call apply_structural_dihedral_about_x
so global matrices, mode vectors, beam nodes, and/or aerogrid are congruent
with a dihedral rotation about +X before aero-structural coupling.

Config flags ``dihedral_rotate_beam`` and ``dihedral_rotate_aerogrid`` select whether
to rotate only the beam model, only the DLM aerogrid, or both (default).
"""

from __future__ import annotations

import numpy as np


def rotate_aerogrid_about_x(aerogrid: dict, dihedral_deg: float) -> dict:
    """Rotate aerogrid point/normal vectors with panelaero_utl convention (rows @ R.T)."""
    from panelaero_utl.rotate_aerogrid import _rotation

    return _rotation(aerogrid, np.deg2rad(float(dihedral_deg)), axis="x")


def apply_structural_dihedral_about_x(
    beam_model,
    structural_results,
    aerogrid,
    dihedral_deg: float,
    *,
    rotate_beam: bool = True,
    rotate_aerogrid: bool = True,
):
    """
    Rotate beam geometry, aerogrid, and/or structural matrices/vectors by dihedral about +X.

    Uses congruence K' = T^T K T, M' = T^T M T with T = I_n ⊗ T6_x so natural
    frequencies are unchanged; mode shapes are expressed in the same global
    DOF basis after the rigid rotation of the model.

    Parameters
    ----------
    rotate_beam : bool
        If True, rotate beam nodes, element K/M, and global structural matrices/modes.
    rotate_aerogrid : bool
        If True, rotate DLM panel aerogrid coordinates and normals.
    structural_results : namedtuple StructuralResults (mutable fields replaced via _replace)
    """
    dihedral_deg = float(dihedral_deg)
    if abs(dihedral_deg) < 1e-12:
        return structural_results, beam_model, aerogrid
    if not rotate_beam and not rotate_aerogrid:
        return structural_results, beam_model, aerogrid

    sr = structural_results

    if rotate_beam:
        from FEA.fea_utl.rotate_beams_x import get_T6_x

        Tx6 = get_T6_x([dihedral_deg])
        R = Tx6[:3, :3]
        n_nodes = len(beam_model["nodes"])
        total_dof = structural_results.total_dof
        if n_nodes * 6 != total_dof:
            raise ValueError("beam_model node count does not match structural_results.total_dof")

        T_full = np.kron(np.eye(n_nodes, dtype=float), Tx6)

        K_g = T_full.T @ structural_results.K_global @ T_full
        M_g = T_full.T @ structural_results.M_global @ T_full

        constrained = np.asarray(structural_results.constrained_dofs, dtype=int)
        free_dofs = np.setdiff1d(np.arange(total_dof), constrained)
        T_ff = T_full[np.ix_(free_dofs, free_dofs)]

        Mff = T_ff.T @ structural_results.Mff @ T_ff
        Kff = T_ff.T @ structural_results.Kff @ T_ff
        Cff = T_ff.T @ structural_results.Cff @ T_ff

        u_full = T_full.T @ structural_results.u_full
        ev_full = T_full.T @ structural_results.dry_eigenvectors_full
        ev_free = ev_full[free_dofs, :]

        for node in beam_model["nodes"]:
            p = np.asarray(node["position"], dtype=float)
            node["position"] = (R @ p).tolist()

        for pm in beam_model.get("point_masses", []):
            off = np.asarray(pm.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            if np.linalg.norm(off) > 1e-15:
                pm["offset"] = tuple((R @ off).tolist())

        from FEA.fea_utl.multibody_assembly import (
            compose_T6_on_beam_model,
            orient_element_section_from_local,
        )

        geo = beam_model.get("geometry", {})
        if geo.get("section_axes_along_global_span", False):
            compose_T6_on_beam_model(beam_model, Tx6)
        else:
            nodes = beam_model["nodes"]
            for elem in beam_model.get("elements", []):
                if "section_stiffness_local" not in elem:
                    continue
                n1, n2 = elem["nodes"]
                p1 = nodes[n1]["position"]
                p2 = nodes[n2]["position"]
                if geo.get("section_axes_chord_locked", False):
                    chord_dir = geo.get("chord_direction_global")
                    thick_dir = geo.get("thickness_direction_global")
                else:
                    chord_dir = elem.get("chord_dir_global")
                    thick_dir = elem.get("thickness_dir_global")
                orient_element_section_from_local(
                    elem,
                    p1,
                    p2,
                    elem["section_stiffness_local"],
                    elem["section_mass_local"],
                    chord_direction_global=chord_dir,
                    thickness_direction_global=thick_dir,
                )

        sr = structural_results._replace(
            K_global=K_g,
            M_global=M_g,
            Mff=Mff,
            Kff=Kff,
            Cff=Cff,
            u_full=u_full,
            dry_eigenvectors_full=ev_full,
            dry_eigenvectors=ev_free,
        )

    if rotate_aerogrid:
        aerogrid = rotate_aerogrid_about_x(aerogrid, dihedral_deg)

    return sr, beam_model, aerogrid
